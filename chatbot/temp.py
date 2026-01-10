import os
import uvicorn
import logging
from contextlib import asynccontextmanager
from typing import Annotated, List, Literal

# --- 第三方库 ---
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

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

# ==========================================
# 1. 配置管理 (使用 Pydantic 强类型校验)
# ==========================================
class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    TAVILY_API_KEY: str
    DB_URI: str  # postgresql://user:pass@host:port/db
    MODEL_NAME: str = "gpt-4o-mini"
    
    # 生产环境建议开启 LangSmith
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_API_KEY: str = ""

    class Config:
        env_file = ".env"

settings = Settings()

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ==========================================
# 2. 数据库连接池 (Global Singleton)
# ==========================================
# 在 FastAPI 生命周期中管理，避免重复创建
connection_pool = AsyncConnectionPool(
    conninfo=settings.DB_URI,
    max_size=20, # 根据你的服务器负载调整
    kwargs={"autocommit": True}
)

# ==========================================
# 3. 定义图 (Graph Logic)
# ==========================================

# 3.1 定义 State
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# 3.2 准备工具
search_tool = TavilySearch(max_results=3)
tools = [search_tool]
tool_node = ToolNode(tools)

# 3.3 模型初始化
llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=0.5, # 生产环境保持适度确定性
).bind_tools(tools)

# 3.4 核心逻辑：消息修剪 (Trim Messages)
# 这是生产环境必须的！防止历史记录过长爆 Token
trimmer = trim_messages(
    max_tokens=2000, # 保留最近 2000 token
    strategy="last",
    token_counter=llm,
    include_system=True, # 始终保留 System Prompt
    allow_partial=False,
    start_on="human",
)

async def chat_node(state: State):
    # 1. 获取所有消息
    messages = state["messages"]
    
    # 2. 应用修剪逻辑 (只把修剪后的消息传给 LLM，但数据库里还是存全量，或者根据需求决定)
    # 这里的 system prompt 最好在 trimmer 之后再次确认存在，或者由 trimmer 自动保留
    trimmed_messages = trimmer.invoke(messages)
    
    # 确保 System Prompt 始终在最前（如果 trimmer 策略比较激进）
    if not isinstance(trimmed_messages[0], SystemMessage):
         trimmed_messages.insert(0, SystemMessage(content="你是一个专业的AI助手，负责处理用户请求并调用工具。"))

    # 3. 调用模型
    response = await llm.ainvoke(trimmed_messages)
    return {"messages": [response]}

# 3.5 构建图
def build_graph():
    workflow = StateGraph(State)
    workflow.add_node("chat_bot", chat_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "chat_bot")
    workflow.add_conditional_edges("chat_bot", tools_condition)
    workflow.add_edge("tools", "chat_bot")

    return workflow

# ==========================================
# 4. FastAPI 服务与生命周期管理
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 启动时 ---
    logger.info("Initializing Database Connection...")
    await connection_pool.open()
    
    # 初始化 Checkpointer 的表结构
    async with connection_pool.connection() as conn:
        checkpointer = AsyncPostgresSaver(conn)
        await checkpointer.setup()
    
    logger.info("System Ready.")
    yield
    # --- 关闭时 ---
    logger.info("Closing Database Connection...")
    await connection_pool.close()

app = FastAPI(title="Production AI Agent", lifespan=lifespan)

# 请求体模型
class ChatRequest(BaseModel):
    message: str
    thread_id: str
    
# ==========================================
# 5. API 接口 (支持流式输出)
# ==========================================

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    生产级对话接口，返回流式响应 (Server-Sent Events 风格)
    """
    checkpointer = AsyncPostgresSaver(connection_pool)
    
    # 编译图
    graph = build_graph().compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": req.thread_id}}
    input_message = HumanMessage(content=req.message)

    async def event_generator():
        try:
            # 使用 astream_events 获取更细粒度的流 (token 级别 + 工具状态)
            # 或者使用 standard stream ("updates")
            async for event in graph.astream(
                {"messages": [input_message]}, 
                config, 
                stream_mode="updates"
            ):
                # 解析事件并格式化为 SSE 格式或 JSON 行
                for node_name, node_value in event.items():
                    if node_name == "chat_bot":
                        latest_msg = node_value["messages"][-1]
                        if latest_msg.content:
                            # 发送文本内容
                            yield f"data: {json.dumps({'type': 'content', 'payload': latest_msg.content}, ensure_ascii=False)}\n\n"
                        if latest_msg.tool_calls:
                            # 通知前端正在调用工具
                            yield f"data: {json.dumps({'type': 'status', 'payload': 'Calling tools...'}, ensure_ascii=False)}\n\n"
                    
                    elif node_name == "tools":
                         yield f"data: {json.dumps({'type': 'status', 'payload': 'Tool execution finished.'}, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            yield f"data: {json.dumps({'type': 'error', 'payload': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ==========================================
# 6. 入口点
# ==========================================
import json

if __name__ == "__main__":
    # 生产环境通常使用 gunicorn 管理 uvicorn workders:
    # gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)