import os
import uvicorn
import logging
from contextlib import asynccontextmanager
from typing import Annotated, List, Literal

# --- 第三方库 ---
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- LangChain / LangGraph ---
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, trim_messages
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver # 注意这里是 aio (Async IO)

# --- 数据库驱动 ---
from psycopg_pool import AsyncConnectionPool


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str
    TAVILY_API_KEY: str
    DB_URI: str
    MODEL_NAME: str = "gpt-4o-mini"
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_API_KEY: str = ""

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        extra = "ignore"
    )

#设置配置
settings = Settings()

#设置日志
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger("app")

#设置连接池，使postgres连接更安全

connection_pool = AsyncConnectionPool(
    conninfo = settings.DB_URI,
    max_size = 20,
    kwargs = {"autocommit": True},
    open = False
)

from typing_extensions import TypedDict

class State(TypedDict):
    messages : Annotated[List[BaseMessage], add_messages]

search_tool = TavilySearch(
    tavily_api_key = settings.TAVILY_API_KEY,
    max_results = 5
)

#初始化工具调用
tools = [search_tool]
tool_node = ToolNode(tools)

#初始化模型
llm = ChatOpenAI(
    model_name = settings.MODEL_NAME,
    api_key = settings.OPENAI_API_KEY,
    base_url = settings.OPENAI_BASE_URL,
    temperature = 0.5
).bind_tools(tools)

#初始化消息裁剪器，防止爆token
trimmer = trim_messages(
    max_tokens = 3000,
    strategy = "last",
    token_counter = llm,
    include_system = False,
    allow_partial = False,
    start_on = "human",
)

#定义聊天节点
async def chat_node(state: State):
    messages = state.get("message", [])
    trimmed_messages = trimmer.invoke(messages)

    if not isinstance(trimmed_messages[0], SystemMessage):
        trimmed_messages.insert(0, SystemMessage(content = "你是一个聊天机器人"))
    
    response = llm.invoke(trimmed_messages)
    return {"messages" : [response.content]}

#定义工作流
def build_graph():
    workflow = StateGraph(State)
    workflow.add_node("chat_node", chat_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "chat_node")
    workflow.add_conditional_edges("chat_node", tools_condition)
    workflow.add_edge("tools", "chat_node")
    workflow.add_edge("chat_node", END)

    return workflow

@asynccontextmanager
async def lifespan(app: FastAPI):
    #启动
    logger.info("Initializing Database Connection...")
    await connection_pool.open()

    async with connection_pool.connection() as conn:
        checkpointer = AsyncPostgresSaver(conn)
        await checkpointer.setup()

    logger.info("System Ready.")

    yield

    #关闭
    logger.info("Closing Dadabase Connection...")
    await connection_pool.close()

app = FastAPI(
    title = "Chat bot agent",
    lifespan = lifespan
)

#定义请求体
class ChatRequest(BaseModel):
    message: str
    thread_id: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    checkpointer = AsyncPostgresSaver(connection_pool)

    graph = build_graph().compile(checkpointer = checkpointer)

    config = {"configurable" : {
        "thread_id" : req.thread_id,
    }}

    input_message = HumanMessage(content = req.message)

    async def event_generator():
        try:
            async for event in graph.astream(
                {"messages" : [input_message]},
                config,
                stream_mode = "updates"
            ):
                for node_name, node_value in event.items():
                    if node_name == "chat_bot":
                        message = node_value.get("message", [])
                        if not message:
                            continue
                        latest_msg = message[-1]
                        if latest_msg.content:
                            yield f"data: {json.dumps({'type' : 'content', 'payload' : latest_msg.content}, ensure_ascii = False)}\n\n"
                        if latest_msg.tool_calls:
                            yield f"data: {jsondumps({'type' : 'status', 'payload' : 'Calling tools...'}, ensure_ascii = False)}\n\n"
                    elif node_name == "tools":
                        yield f"data: {json.dumps({'type' : 'status', 'payload' : 'Tool execution finished.'}, ensure_ascii = False)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            yield f"data: {json.dumps({'type' : 'error', 'payload' : str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type = "text/event-stream")

import json

if __name__ == "__main__":

    uvicorn.run(
        "chat_agent:app",
        host = "0.0.0.0",
        port = 8000,
        reload = True
    )