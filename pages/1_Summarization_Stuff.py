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
st.header("🤖 PDF Summarization (Stuff)")
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


llm = load_llm(api_provider, api_key, model_name)

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


doc = load_and_collapse_pdf(uploaded_pdf)


# When code is conditioned on a button's value, it will execute once in response to the button being clicked
# and not again (until the button is clicked again).
start_summarize = st.button("summarize", type="primary", disabled=not llm)
if not start_summarize:
    st.info('Click on summarize', icon="🚨")
    st.stop()


token_number = get_context_length(doc)

if token_number > context_length_limit:
    st.info("The number of tokens in PDF text is {}, which is more than {}".format(token_number, context_length_limit), icon="🚨")
    st.stop()


chain = load_chain(llm)

container = st.container(border=True)
container.write("Summary:")

if stream_enabled:
    stream = chain.stream({"context": doc.page_content})
    container.write_stream(stream)
else:
    result = chain.invoke({"context": doc.page_content})
    container.write(result.content)


st.success("Summarization finished!", icon="✅")

