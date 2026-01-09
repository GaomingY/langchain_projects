from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
load_dotenv(override = True)

class State(TypedDict):
    messages : Annotated[list, add_messages]

def create_chat_agent(model_name):
    llm = ChatOpenAI(
        model_name = model_name,
        api_key = os.getenv("OPENAI_API_KEY"),
        base_url = os.getenv("OPENAI_BASE_URL"),
        temperature = 1.0,
    )
    
    def chat_bot_chain(state):
        prompt = [SystemMessage(content = "你是一个聊天机器人")] + state["messages"]
        return {"messages" : [llm.invoke(prompt)]}
    
    workflow = StateGraph(State)
    workflow.add_node("chat_bot", chat_bot_chain)
    workflow.add_edge(START, "chat_bot")
    workflow.add_edge("chat_bot", END)
    return workflow.compile()

def stream_graph_updates(graph, user_input: str):
    for event in graph.stream(
        {
            "messages" : [HumanMessage(content = user_input)]
        }
    ):
        for value in event.values():
                print("Chat_bot:", value["messages"][-1].content)

if __name__ == "__main__":
    chat_bot = create_chat_agent("gpt-4o-mini")

    while True:
        user_input = input("User:")
        if user_input.lower() in ["quit", "q", "exit"]:
            print("goodbye")
            break
        stream_graph_updates(chat_bot, user_input)