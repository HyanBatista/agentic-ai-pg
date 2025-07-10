import streamlit as st
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Please respond to the question asked.",
        ),
        ("user", "Question: {question}"),
    ]
)

st.title("LangChain Demo with Llama2")
input_text = st.text_input("What question do you have in mind?")
llm = Ollama(model="gemma3")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
