from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_core.runnables.graph import Node
from langchain_openai import ChatOpenAI
import os
llm = ChatOpenAI(
    model_name = "gpt-4o-mini",
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url = os.getenv("OPENAI_BASE_KEY"),
    temperature = 1.0,
)

system_prompt_gather = """Your job is to get information from a user about what type of prompt template they want to create.

You should get the following information from them:

- What the objective of the prompt is
- What variables will be passed into the prompt template
- Any constraints for what the output should NOT do
- Any requirements that the output MUST adhere to

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

After you are able to discern all the information, call the relevant tool."""

from langchain_core.messages import SystemMessage
from pydantic import BaseModel
from typing import List

def get_messages_info(messages):
    return [SystemMessage(content = system_prompt_gather)] + messages

def PromptInstructions(BaseModel):
    objective : str
    variables : List[str]
    constraints : List[str]
    requirements : List[str]

llm_with_tool = llm.bind_tools([PromptInstructions])

def info_chain(state):
    messages = get_messages_info(state["messages"])
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}

system_template_gen = """Baseed on the following requirements, write a good prompt template"""

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def get_gen_messages(messages):
    tool_call = None
    other_msgs = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call = m.tool_calls[0]["args"]
        elif isinstance(ToolMessage):
            continue
        elif tool_call is not None:
            other_msgs.append(m)

    return [SystemMessage(content = system_template_gen.format(reqs=tool_call))] + other_msgs

def gen_chain(state):
    gen_messages = get_gen_messages(state["messages"])
    response = llm_with_tool.invoke(gen_messages)
    return {"messages" : [response]}

from typing import Literal
from langgraph.graph import END

def get_state(state):
    messages = state["messages"]

    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    # llm的信息但没有调用工具，说明AI回答已经结束，工作流也应当结束
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "info"

# 创建工作流，并编译得到图
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    messages: Annotated[list, add_messages]


memory = InMemorySaver()
workflow = StateGraph(State)
workflow.add_node("info", info_chain)
workflow.add_node("generate", gen_chain)

@workflow.add_node
def add_tool_message(state : State):
    return {
        "messages":[
            ToolMessage(
                content = "Prompt generated!",
                tool_call_id = state["messages"][-1].tool_calls[0]["id"],
            )
        ]
    }

workflow.add_conditional_edges("info", get_state, ["add_tool_message", "info", END])
workflow.add_edge(START, "info")
workflow.add_edge("add_tool_message", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile(checkpointer = memory)

# 得到图（即定义了agent的处理逻辑）之后，输入用户请求运行chatbot
import uuid

cached_human_responses = ["hi!", "rag prompt", "1 rag, 2 none, 3 no, 4 no", "red", "q"]
cached_response_index = 0
config = {"configurable" : {"thread_id": str(uuid.uuid4())}}

while True:
    try:
        user = input("User (q/Q to quit): ")
    except:
        user = cached_human_responses[cached_response_index]
        cached_response_index += 1

    if user in {"q", "Q"}:
        print("AI: Byebye")
        break

    output = None
    for output in graph.stream(
        {"messages": [HumanMessage(content = user)]}, config = config, stream_mode = "updates"
    ):
        last_message = next(iter(output.values()))["messages"][-1]
        last_message.pretty_print()
    
    if output and "generate" in output:
        print("Done!")
    