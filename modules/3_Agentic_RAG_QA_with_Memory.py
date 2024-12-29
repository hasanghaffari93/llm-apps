import os
import uuid
import tempfile
from typing_extensions import TypedDict, List
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages, RemoveMessage
from utils import get_context_length_limit, update_api_and_model, authenticate, stream_enabled, select_embedding_model
import faiss


@st.cache_resource(max_entries=1)
def get_vector_store(uploaded_pdf):

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
def app(model):
    # https://python.langchain.com/docs/tutorials/qa_chat_history/

    Chat_Model = ChatOpenAI if st.session_state.api == "OpenAI" else ChatGroq
    llm = Chat_Model(model=model, api_key=st.session_state.api_key)    

    graph_builder = StateGraph(MessagesState)


    # return a tuple of (content, artifact)
    # `content` is fed back to a the model
    # `artifact` is accessible to downstream components in our chain or agent
    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        # a docstring MUST be provided
        """Retrieve information related to a query."""
        retrieved_docs = vector_store.similarity_search(query, k=5)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    # Step 1: Generate an AIMessage that `may` include a tool-call to be sent.
    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}


    # Step 2: Execute the retrieval.
    # 'ToolNode' is  is a LangChain Runnable.
    # This node takes graph state (with a list of messages) as input 
    # and outputs state update with the result of tool calls. 
    # It can work with any StateGraph as long as its state has a messages key 
    # with an appropriate reducer (see MessagesState).
    tools = ToolNode([retrieve])


    # Step 3: Generate a response using the retrieved content.
    def generate(state: MessagesState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = llm.invoke(prompt)
        return {"messages": [response]}

    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition, # Prebuilt Method: Use in the conditional_edge to route to the ToolNode if the last message has tool calls. Otherwise, route to the end.
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph

def clear_chat_history(chatbot, config):
    messages = chatbot.get_state(config).values["messages"]
    for message in messages:
        st.write(message.type)
    chatbot.update_state(config, {"messages": [RemoveMessage(id=m.id) for m in messages]}, as_node="__start__")  


# UI Setup
st.header(":material/docs: Agentic RAG: QA with Memory")

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

continue_run = st.session_state["valid_auth"] and uploaded_pdf

if continue_run:
    vector_store = get_vector_store(uploaded_pdf)
    graph = app(model)

config = {"configurable": {"thread_id": "abc123"}}


if st.sidebar.button("New Chat", disabled=not st.session_state["valid_auth"]):
    clear_chat_history(graph, config)

# Show chat History
if continue_run and "messages" in graph.get_state(config).values:
    for message in graph.get_state(config).values["messages"]:
        if message.type == "human" or (message.type == "ai" and not message.tool_calls):
            st.chat_message(message.type).write(message.content)

def stream_wrapper(stream):
    for message, _ in stream:
        if message.type == "AIMessageChunk" and not message.tool_calls:
            yield message.content

if input_message := st.chat_input(disabled=not continue_run):
    st.chat_message(name="human").write(input_message) 

    if not is_stream:
        response = graph.invoke(
            {"messages": [{"role": "user", "content": input_message}]},
            config=config,
        )
        st.chat_message(name="assistant").write(response["messages"][-1].content)

    else:
        stream = graph.stream(
            {"messages": [{"role": "user", "content": input_message}]},
            stream_mode="messages",
            config=config,
        )
        st.chat_message(name="assistant").write(stream_wrapper(stream))

        