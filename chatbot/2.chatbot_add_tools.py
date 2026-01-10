from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI
import os
import json
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
load_dotenv()

search_tool = TavilySearch(max_results = 2)
tools = [search_tool]

class State(TypedDict):
    messages: Annotated[list, add_messages]

class BasicToolNode:

    def __init__(self, tools: list) -> None:
        self.tool_by_name = {tool.name : tool for tool in tools}
    def __call__(self, state):
        if messages := state.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tool_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content = json.dumps(tool_result),
                    name = tool_call["name"],
                    tool_call_id = tool_call["id"],
                )
            )
        return {"messages": outputs}
    
def create_chat_bot_with_tools(model_name) -> StateGraph:
    system_prompt = SystemMessage(content = "你是一个聊天机器人")

    llm = ChatOpenAI(
        model_name = model_name,
        api_key = os.getenv("OPENAI_API_KEY"),
        base_url = os.getenv("OPENAI_BASE_URL"),
        temperature = 1.0,
    ).bind_tools(tools)

    def chat_bot_node(state):
        prompt = [system_prompt] + state["messages"]
        return {"messages": [llm.invoke(prompt)]}
    
    workflow = StateGraph(State)
    workflow.add_node("chat_bot", chat_bot_node)
    workflow.add_edge(START, "chat_bot")
    tool_node = BasicToolNode(tools = tools)
    workflow.add_node("tools", tool_node)
    workflow.add_conditional_edges("chat_bot", route_tools, {"tools": "tools", END: END})
    workflow.add_edge("tools", "chat_bot")
    return workflow.compile()

def route_tools(
    state: State,
):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError("No messsage is found")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages" : [
            HumanMessage(content = user_input),
        ]}
    ):
        for value in event.values():
            # value is the state after each node finishes
            if "messages" in value and value["messages"]:
                print("Chat_bot: ", value["messages"][-1].content)

if __name__ == "__main__":
    graph = create_chat_bot_with_tools("gpt-4o-mini")
    
    while True:
        try:
            user_input = input("Usr: ")
            if(user_input.lower() in ["q", "quit", "exit"]):
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except Exception as e:
            raise RuntimeError(f"运行失败: {e}") from e