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


class AppConfig:
    API_PROVIDERS = ('OpenAI', 'Groq')
    MODEL_NAMES = {
        'OpenAI': ('gpt-4o-mini', 'gpt-4-turbo', 'gpt-4o'),
        'Groq': ('llama3-70b-8192', 'llama3-8b-8192'),
    }
    BASE_URLS = {
        'OpenAI': 'https://api.openai.com/v1',
        'Groq': 'https://api.groq.com/openai/v1',
    }
    KEY_NAMES = {
        'OpenAI': 'OPENAI_API_KEY_DEV',
        'Groq': 'GROQ_API_KEY_DEV',
    }
    DEFAULT_MAX_TOKENS = 3000
    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

@st.cache_data()
def validate_api_key(api_key: str, api_provider: str) -> bool:
    try:
        client = openai.OpenAI(
            base_url=AppConfig.BASE_URLS[api_provider],
            api_key=api_key,
        )
        client.models.list()
        return True
    except openai.AuthenticationError:
        return False
    except Exception as error:
        st.sidebar.error(str(error))
        return False

# UI Setup
st.header("🤖 Large Document Summarization (Map-Reduce)")
st.caption("🚀 Powered by LangChain | 🔥 OpenAI & Groq LLMs | 🛠️ Streamlit UI")

# Sidebar Configuration
api_provider = st.sidebar.selectbox(
    label="Which API do you want to use?",
    options=AppConfig.API_PROVIDERS,
    index=0,
    key="api_provider",
    placeholder="Select an API...",
    # on_change=restart_chat,
)

model_name = st.sidebar.selectbox(
    label="Which model do you want to use?",
    options=AppConfig.MODEL_NAMES[api_provider],
    index=0,
    key="model_name",
    placeholder="Select a model...",
    # on_change=restart_chat,
)

context_length_limit = st.sidebar.number_input(
    label="Context length limit:",
    min_value=10,
    max_value=128000,
    value=AppConfig.DEFAULT_MAX_TOKENS,
    step=1000,
    key="context_length_limit", 
    help="Long context costs a lot!",
    placeholder="3000",
    # on_change=restart_chat,
)

stream_enabled = st.sidebar.checkbox(
    label="Enable stream chat",
    value=True,
    key="stream",
    help="The output will be streaming",
)

st.sidebar.divider()

auth_type = st.sidebar.radio(
    "How would you like to authenticate?",
    ["Use an API Key", "Use a Password"],
)

if auth_type == "Use an API Key":
    label = "{} API Key:".format(api_provider)
else:
    label = "Password:"

auth = st.sidebar.text_input(
    label=label,
    key="auth",
    type="password",
)

st.session_state["valid_auth"] = False

api_key = ""
if auth_type == "Use an API Key":
    if validate_api_key(auth, api_provider):
        api_key = auth
        st.session_state["valid_auth"] = True
        st.sidebar.success("Valid API key")
    else:
        st.sidebar.error("Invalid API key")
else:
    if auth in st.secrets["PASSWORDS"]:
        api_key = st.secrets[AppConfig.KEY_NAMES[api_provider]]
        st.session_state["valid_auth"] = True
        st.sidebar.success("Valid password")
    else:
        st.sidebar.error("Invalid password")


@st.cache_resource
def load_llm(api_provider: str, api_key: str, model_name: str):

    if not st.session_state["valid_auth"]:
        return None

    llm = ChatOpenAI(model=model_name, api_key=api_key) if api_provider == "OpenAI" \
        else ChatGroq(model=model_name, api_key=api_key)
    
    return llm


def combine_documents(docs: List[Document]) -> str:
    return " ".join(doc.page_content for doc in docs)


def get_context_length(doc):
    return llm.get_num_tokens(doc.page_content)
    

@st.cache_data
def load_and_split_pdf(pdf_file):

    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, pdf_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(pdf_file.getvalue())


    loader = PyPDFLoader(temp_filepath)

    # Ir returns `List(Document)`, each `Document` corresponds to a page of PDF
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0  # TODO?? ## Chunk size?
    )
    doc_splits = text_splitter.split_documents(docs)

    return doc_splits


llm = load_llm(api_provider, api_key, model_name)


@st.cache_resource
def app():
    return load_agent()

config = {"recursion_limit": 10,
            "configurable": {"llm": llm,
                            "token_max": 1000}}

async def summarize():

    app = load_agent()

    if stream_enabled:
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


uploaded_pdf = st.file_uploader(label='Choose your .pdf file',
                                 type="pdf",
                                 accept_multiple_files =False,
                                 key="uploaded_pdf",
                                )

if not uploaded_pdf:
    # Display an informational message.
    st.info('Please upload a PDF document to continue', icon="🚨")
    # Stops execution immediately.
    st.stop()


doc_splits = load_and_split_pdf(uploaded_pdf)


# When code is conditioned on a button's value, it will execute once in response to the button being clicked
# and not again (until the button is clicked again).
start_summarize = st.button("summarize", type="primary", disabled=not llm)
if not start_summarize:
    st.info('Click on summarize', icon="🚨")
    st.stop()


with st.spinner('Summarizing...'):
    container = st.container(border=True)
    container.write("Summary:")
    asyncio.run(summarize())

st.success("Summarization finished!", icon="✅")
