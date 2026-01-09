from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, ToolMessage
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
    message : Annotated[list, add_messages]

class BasicToolNode:

    def __init__(self, tools: list) -> None:
        self.tool_by_name = {tool.name : tool for tool in tools}
    def __call__(self, state):
        if messages := state.get("message", []):
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
                    tool_cal_id = tool_call["id"],
                )
            )
        return {"messages", outputs}
    
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
        return {"messages" : llm.invoke(prompt)}
    
    workflow = StateGraph(State)
    workflow.add_node("chat_bot", chat_bot_node)
    workflow.add_edge(START, "chat_bot")
    workflow.add_edge("chat_bot", END)
    tool_node = BasicToolNode(tools = tools)
    workflow.add_node("tools", tool_node)
    return workflow.compile()


if __name__ == "__main__":
    graph = create_chat_bot_with_tools("gpt-4o-mini")
    png_data = graph.get_graph().draw_mermaid_png()
    with open("graph_with_tools.png", "wb") as f:
        f.write(png_data)
