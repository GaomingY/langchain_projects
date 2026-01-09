from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv(override = True)

model_name = "gpt-4o-mini"
base_llm = ChatOpenAI(
    model_name = model_name,
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url = os.getenv("OPENAI_BASE_URL"),
    temperature = 1.0,
)

