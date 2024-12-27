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
from utils import get_context_length_limit, update_api_and_model, authenticate, stream_enabled, select_embedding_model


@st.cache_resource(max_entries=1)
def load_llm(api: str, api_key: str, model: str):
    return ChatOpenAI(model=model, api_key=api_key) if api == "OpenAI" \
        else ChatGroq(model=model, api_key=api_key)


@st.cache_resource(max_entries=1)
def load_embedding(api_key: str, embedding_model: str):

    if not st.session_state["valid_auth"]:
        return None

    embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=api_key)
    
    return embeddings

@st.cache_resource(max_entries=1)
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
        model=st.session_state["embedding_model"], 
        openai_api_key=st.session_state["api_key"]
    )

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

@st.cache_resource(max_entries=1)
def app():

    llm = load_llm(st.session_state.api, st.session_state.api_key, model)

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

def stream_wrapper(stream):
    for chunk, _ in stream:
        yield chunk.content

# UI Setup
st.header("🤖 RAG")
st.caption("🚀 Powered by LangChain | 🔥 OpenAI & Groq LLMs | 🛠️ Streamlit UI")

# Sidebar
model = update_api_and_model("OpenAI")
select_embedding_model()
authenticate()
context_length_limit = get_context_length_limit()
is_stream = stream_enabled()

uploaded_pdf = st.file_uploader(
    label='Choose your .pdf file',
    type="pdf",
    accept_multiple_files =False,
    disabled=not st.session_state["valid_auth"]
)

if not uploaded_pdf:
    st.info("Please upload a PDF document to continue.")
    st.stop()

vector_store = configure_retriever(uploaded_pdf)

graph = app()
    
if question := st.chat_input(disabled=not st.session_state["valid_auth"]):
    st.chat_message(name="human").write(question)

    if not is_stream:
        response = graph.invoke({"question": question})
        st.chat_message(name="assistant").write(response["answer"])
    else:
        stream = graph.stream({"question": question}, stream_mode="messages")
        st.chat_message(name="assistant").write(stream_wrapper(stream))
