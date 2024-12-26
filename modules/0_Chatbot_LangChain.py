from typing import Annotated, Sequence, Dict
from typing_extensions import TypedDict
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, BaseMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages, RemoveMessage
from utils import get_max_tokens, get_system_prompt, select_api_and_model, authenticate, stream_enabled

def restart_chat():
    if chatbot and chatbot.get_state(config).values:
        messages = chatbot.get_state(config).values["messages"]
        chatbot.update_state(config, {"messages": [RemoveMessage(id=m.id) for m in messages]})     

@st.cache_resource
def create_chatbot(api: str, api_key: str, model: str, max_tokens: int):

    if not st.session_state["valid_auth"]:
        return None

    llm = ChatOpenAI(model=model, api_key=api_key) if api == "OpenAI" \
        else ChatGroq(model=model, api_key=api_key)

    trimmer = trim_messages(
        max_tokens=max_tokens,
        strategy="last",
        token_counter=llm,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]    

    workflow = StateGraph(state_schema=State)

    def call_model(state: State) -> Dict[str, list]:
        trimmed_messages = trimmer.invoke(state["messages"])
        prompt = prompt_template.invoke({"messages": trimmed_messages})
        response = llm.invoke(prompt)
        return {"messages": [response]}

    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    return workflow.compile(checkpointer=MemorySaver())


def stream_wrapper(stream):
    for chunk, _ in stream:
        yield chunk.content   

# UI Setup
st.header("🤖 Chatbot")
st.caption("🚀 Powered by LangChain | 🔥 OpenAI & Groq LLMs | 🛠️ Streamlit UI")
get_system_prompt(restart_chat)


# Sidebar
select_api_and_model(on_change=restart_chat)
authenticate()
get_max_tokens()
stream_enabled()


prompt_template = ChatPromptTemplate.from_messages([
    ("system", st.session_state["system_prompt"]),
    MessagesPlaceholder(variable_name="messages"),
])

chatbot = create_chatbot(
    st.session_state["api"], 
    st.session_state["api_key"], 
    st.session_state["model"], 
    st.session_state["max_tokens"]
)

config = {"configurable": {"thread_id": "abc345"}}

# Show chat History
if chatbot and "messages" in chatbot.get_state(config).values:
    for msg in chatbot.get_state(config).values["messages"]:
        st.chat_message(msg.type).write(msg.content)

# Chat Interface
if query := st.chat_input(disabled=not st.session_state["valid_auth"]):
    st.chat_message(name="human").write(query)
    input_messages = [HumanMessage(query)]

    if not st.session_state["stream"]:
        output = chatbot.invoke({"messages": input_messages}, config)
        st.chat_message(name="assistant").write(output["messages"][-1].content)
    else:
        output = chatbot.stream({"messages": input_messages}, config, stream_mode="messages")
        st.chat_message(name="assistant").write(stream_wrapper(output))

