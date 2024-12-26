import os
import tempfile
import streamlit as st
import openai
from typing import List, Literal
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.reduce import collapse_docs
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from utils import get_context_length_limit, select_api_and_model, authenticate, stream_enabled


@st.cache_resource
def load_llm(api: str, api_key: str, model: str):

    if not st.session_state["valid_auth"]:
        return None

    llm = ChatOpenAI(model=model, api_key=api_key) if api == "OpenAI" \
        else ChatGroq(model=model, api_key=api_key)
    
    return llm


def load_chain(llm):
    template = "Write a concise summary of the following:\\n\\n{context}"
    prompt = ChatPromptTemplate([("system", template)])

    chain = prompt | llm
    return chain


def combine_documents(docs: List[Document]) -> str:
    return " ".join(doc.page_content for doc in docs)


def get_context_length(doc):
    return llm.get_num_tokens(doc.page_content)
    

@st.cache_data
def load_and_collapse_pdf(pdf_file):

    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, pdf_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(pdf_file.getvalue())

    loader = PyPDFLoader(temp_filepath)

    # Ir returns `List(Document)`, each `Document` corresponds to a page of PDF
    docs = loader.load()

    # It just merges the metadatas of a set of documents after executing a collapse function on them. 
    doc = collapse_docs(docs=docs, combine_document_func=combine_documents)

    return doc


# UI Setup
st.header("🤖 PDF Summarization (Stuff)")
st.caption("🚀 Powered by LangChain | 🔥 OpenAI & Groq LLMs | 🛠️ Streamlit UI")

# Sidebar
select_api_and_model()
authenticate()
st.sidebar.divider()
get_context_length_limit()
stream_enabled()

llm = load_llm(
    st.session_state["api"], 
    st.session_state["api_key"], 
    st.session_state["model"]
)

uploaded_pdf = st.file_uploader(
    label='Choose your .pdf file',
    type="pdf",
    accept_multiple_files=False,
    key="uploaded_pdf",
)

if not uploaded_pdf:
    st.info('Please upload a PDF document to continue', icon="🚨")
    st.stop() # Stops execution immediately


doc = load_and_collapse_pdf(uploaded_pdf)


# When code is conditioned on a button's value, it will execute once in response to the button being clicked
# and not again (until the button is clicked again).
start_summarize = st.button("summarize", type="primary", disabled=not llm)
if not start_summarize:
    st.info('Click on summarize', icon="🚨")
    st.stop()


token_number = get_context_length(doc)

if token_number > st.session_state["context_length_limit"]:
    st.info("The number of tokens in PDF text is {}, which is more than {}".format(token_number, st.session_state["context_length_limit"]), icon="🚨")
    st.stop()


chain = load_chain(llm)

container = st.container(border=True)
container.write("Summary:")

if st.session_state["stream"]:
    stream = chain.stream({"context": doc.page_content})
    container.write_stream(stream)
else:
    result = chain.invoke({"context": doc.page_content})
    container.write(result.content)

st.success("Summarization finished!", icon="✅")
