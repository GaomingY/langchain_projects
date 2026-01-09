import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override = True)

llm_name = "gpt-4o-mini"
llm = ChatOpenAI(
    model_name = llm_name,
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url = os.getenv("OPENAI_BASE_URL"),
    temperature = 1.0,
)

# 读取langchain官网的LCEL文档作为agent的知识库
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

url = "https://python.langchain.com/docs/concepts/lcel/"
loader = RecursiveUrlLoader(
    url = url, max_depth = 20, extractor = lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

d_sorted = sorted(docs, key = lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)

# 定义prompt模板
from langchain_core.prompts import ChatPromptTemplate

code_gen_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant with expertise in LCEL, LangChain expression language. \n 
    Here is a full set of LCEL documentation:  \n ------- \n  {context} \n ------- \n Answer the user 
    question based on the above provided documentation. Ensure any code you provide can be executed \n 
    with all required imports and variables defined. Structure your answer with a description of the code solution. \n
    Then list the imports. And finally list the functioning code block. Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)