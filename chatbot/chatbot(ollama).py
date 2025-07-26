from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACKING_V2"] = 'true'
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can answer questions and help with user queries."),
        ("user", "Question: {question}")
    ]
)

st.title('Langchain Demo With Ollama')
input_text = st.text_input("Search the topic you want")

model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
llm = Ollama(model=model_name) 
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write("Generating answer...")
    answer = chain.invoke({"question": input_text})
    st.write(answer)
