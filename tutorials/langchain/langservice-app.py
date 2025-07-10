import os

from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langserve import add_routes
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "LangServe"
    debug: bool = False
    port: int = 8000

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    langchain_api_key: str = Field(..., alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(..., alias="LANGCHAIN_PROJECT")
    langchain_tracing_v2: str = Field(..., alias="LANGCHAIN_TRACING_V2")
    hf_token: str = Field(..., alias="HF_TOKEN")
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    user_agent: str = Field(..., alias="USER_AGENT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

os.environ["GROQ_API_KEY"] = settings.groq_api_key

model = ChatGroq(model="gemma-7b-it", temperature=0)  # example model
template = ChatPromptTemplate.from_messages(
    [
        ("system", ""),
        ("user", "{text}"),
    ]
)
parser = StrOutputParser()
chain = template | model | parser

from langserve import validation

validation.BatchRequestShallowValidator.model_rebuild()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="A simple API server using LangChain runnable interfaces.",
)

add_routes(app, chain, path="/chain")
