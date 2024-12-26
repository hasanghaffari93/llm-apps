import os
import uuid
import tempfile
from typing_extensions import TypedDict, List
import streamlit as st
import openai
import faiss
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langgraph.graph import START, StateGraph


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
    EMBEDDING_MODEL_NAMES = {
        'OpenAI': ('text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large'),
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
st.header("🤖 RAG")
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

embedding_model_name = st.sidebar.selectbox(
    label="Which embedding model from OpenAI do you want to use?",
    options=AppConfig.EMBEDDING_MODEL_NAMES["OpenAI"],
    index=0,
    key="embedding_model_name",
    help="This app has API key of these model",
    placeholder="Select an embedding model...",
    disabled=False,
    label_visibility="visible",  #"visible", "hidden", or "collapsed"
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



llm = load_llm(api_provider, api_key, model_name)


uploaded_pdf = st.file_uploader(label='Choose your .pdf file',
                                 type="pdf",
                                 accept_multiple_files =False,
                                 key="uploaded_pdf",
                                )


if not uploaded_pdf:
    st.info("Please upload a PDF document to continue.")
    st.stop()


        

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_pdf):
    # Read documents

    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_pdf.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_pdf.getvalue())
    loader = PyPDFLoader(temp_filepath)
    docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=800,
        chunk_overlap=50,
        length_function=len,
    )
    all_splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model=embedding_model_name, 
        openai_api_key=api_key) ##########################################



    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    
    uuids = [str(uuid.uuid4()) for _ in range(len(all_splits))]

    vector_store.add_documents(documents=all_splits, ids=uuids)

    return vector_store

vector_store = configure_retriever(uploaded_pdf)


@st.cache_resource
def app():

    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    Question: {question}

    Context: {context}

    Answer:"""

    prompt_template = PromptTemplate.from_template(template)    

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    graph_builder = StateGraph(state_schema=State)

    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"], k=2)
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt_template.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}
    
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)

    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")

    app = graph_builder.compile()

    return app

graph = app()

def stream_wrapper(stream):
    for chunk, _ in stream:
        yield chunk.content
    
if question := st.chat_input(disabled=not st.session_state["valid_auth"]):
    st.chat_message(name="human").write(question)

    if not stream_enabled:
        response = graph.invoke({"question": question})
        st.chat_message(name="assistant").write(response["answer"])
    else:
        stream = graph.stream({"question": question}, stream_mode="messages")
        st.chat_message(name="assistant").write(stream_wrapper(stream))
