from dotenv import load_dotenv
import os
import json
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()

memory = InMemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

class BasicToolNode():
    def __init__(self, tools: list) -> None:
        self.tool_by_name = {tool.name : tool for tool in tools}
    def __call__(self, state):
        if messages := state.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tool_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

search_tool = TavilySearch(max_results = 3)
tools = [search_tool]
tool_node = BasicToolNode(tools = tools)
system_prompt = SystemMessage(content = "你是一个聊天机器人")

def create_chat_bot_with_tools_and_memory(model_name: str):

    llm = ChatOpenAI(
        model_name = model_name,
        api_key = os.getenv("OPENAI_API_KEY"),
        base_url = os.getenv("OPENAI_BASE_URL"),
        temperature = 1.0,
    ).bind_tools(tools)

    def chat_bot_node(state):
        prompt = [system_prompt] + state.get("messages", [])
        return {"messages": [llm.invoke(prompt)]}

    def route_tools(state):
        messages = state.get("messages", [])
        if not messages:
            return END
        ai_message = messages[-1]
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return END

    workflow = StateGraph(State)
    workflow.add_node("chat_bot", chat_bot_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "chat_bot")
    workflow.add_conditional_edges("chat_bot", route_tools, ["tools", END])
    workflow.add_edge("tools", "chat_bot")
    
    return workflow.compile(checkpointer = memory)

def stream_graph(user_input: str):
    for nodes in graph.stream(
        {
            "messages" : [HumanMessage(content = user_input)],
        },
        config,
        stream_mode="values",
    ):
        # stream_mode="values" yields a dict-like state per step
        if isinstance(nodes, dict) and nodes.get("messages"):
            print("Chatbot:", nodes["messages"][-1].content)

if __name__ == "__main__":
    model_name = "gpt-4o-mini"
    graph = create_chat_bot_with_tools_and_memory(model_name)
    config = {"configurable" : {"thread_id" : "1"}}

    while True:
        try:
            user_input = input("User: ")
            if(user_input.lower() in ["q", "quit", "exit"]):
                print("Goodbye!")
                break
            stream_graph(user_input)
        except Exception as e:
            raise RuntimeError(f"运行失败: {e}") from e
