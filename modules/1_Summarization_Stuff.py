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
from utils import get_context_length_limit, update_api_and_model, authenticate, stream_enabled


@st.cache_resource(max_entries=1)
def load_llm(api: str, api_key: str, model: str):
    return ChatOpenAI(model=model, api_key=api_key) if api == "OpenAI" \
        else ChatGroq(model=model, api_key=api_key)

def load_chain(llm):
    template = "Write a concise summary of the following:\\n\\n{context}"
    prompt = ChatPromptTemplate([("system", template)])

    chain = prompt | llm
    return chain


def combine_documents(docs: List[Document]) -> str:
    return " ".join(doc.page_content for doc in docs)


def get_context_length(doc, llm):
    return llm.get_num_tokens(doc.page_content)
    

@st.cache_data(max_entries=1)
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
st.header(":material/summarize: PDF Summarization (Stuff)")
st.caption("🚀 Powered by LangChain | 🔥 OpenAI & Groq LLMs | 🛠️ Streamlit UI")

# Sidebar
model = update_api_and_model()
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


start_summarize = st.button("summarize", type="primary", disabled=not st.session_state["valid_auth"])
if not start_summarize:
    st.info('Click on summarize', icon="🚨")
    st.stop()

doc = load_and_collapse_pdf(uploaded_pdf)

llm = load_llm(st.session_state.api, st.session_state.api_key, model)

token_number = get_context_length(doc, llm)

if token_number > context_length_limit:
    st.info("The number of tokens in PDF text is {}, which is more than {}".format(token_number, st.session_state["context_length_limit"]), icon="🚨")
    st.stop()

chain = load_chain(llm)

container = st.container(border=True)
container.write("Summary:")

if is_stream:
    stream = chain.stream({"context": doc.page_content})
    container.write_stream(stream)
else:
    result = chain.invoke({"context": doc.page_content})
    container.write(result.content)

st.success("Summarization finished!", icon="✅")
