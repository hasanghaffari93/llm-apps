import asyncio
import os
import tempfile
import streamlit as st
import openai
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from agents.summarizer import load_agent
from utils import get_context_length_limit, update_api_and_model, authenticate, stream_enabled


@st.cache_resource(max_entries=1)
def load_llm(api: str, api_key: str, model: str):
    return ChatOpenAI(model=model, api_key=api_key) if api == "OpenAI" \
        else ChatGroq(model=model, api_key=api_key)


def combine_documents(docs: List[Document]) -> str:
    return " ".join(doc.page_content for doc in docs)


def get_context_length(doc):
    return llm.get_num_tokens(doc.page_content)
    

@st.cache_data(max_entries=1)
def load_and_split_pdf(pdf_file):

    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, pdf_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(pdf_file.getvalue())


    loader = PyPDFLoader(temp_filepath)

    # It returns `List(Document)`, each `Document` corresponds to a page of PDF
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0  # TODO?? ## Chunk size?
    )
    doc_splits = text_splitter.split_documents(docs)

    return doc_splits

async def summarize(is_stream):

    app = load_agent()

    if is_stream:
        stream = app.astream(
        {"contents": [doc.page_content for doc in doc_splits]},
        config=config,
        )

        # TODO: Update 
        async for step in stream:
            container.write(list(step.keys()))

        container.write(step["generate_final_summary"]['final_summary'])

    else:

        result = await app.ainvoke(
            {"contents": [doc.page_content for doc in doc_splits]},
            config=config,
        )
        container.write(result["final_summary"])

# UI Setup
st.header("🤖 Large Document Summarization (Map-Reduce)")
st.caption("🚀 Powered by LangChain | 🔥 OpenAI & Groq LLMs | 🛠️ Streamlit UI")

# Sidebar
model = update_api_and_model() # Run on_change then rerun!
authenticate()
context_length_limit = get_context_length_limit()
is_stream = stream_enabled()

uploaded_pdf = st.file_uploader(
    label='Choose your .pdf file',
    type="pdf",
    accept_multiple_files=False,
    disabled=not st.session_state["valid_auth"]
)

if not uploaded_pdf:
    st.info('Please upload a PDF document to continue', icon="🚨")
    st.stop()

doc_splits = load_and_split_pdf(uploaded_pdf)


start_summarize = st.button("summarize", type="primary", disabled=not st.session_state["valid_auth"])
if not start_summarize:
    st.info('Click on summarize', icon="🚨")
    st.stop()

llm = load_llm(st.session_state.api, st.session_state.api_key, model)

config = {
    "recursion_limit": 10,
    "configurable": {
        "llm": llm,
        "token_max": st.session_state["context_length_limit"]
    }
}

with st.spinner('Summarizing...'):
    container = st.container(border=True)
    container.write("Summary:")
    asyncio.run(summarize(is_stream))

st.success("Summarization finished!", icon="✅")
