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
from utils import get_max_tokens, get_system_prompt, update_api_and_model, authenticate, stream_enabled

def clear_chat_history(chatbot, config):
    if chatbot and chatbot.get_state(config).values:
        messages = chatbot.get_state(config).values["messages"]
        chatbot.update_state(config, {"messages": [RemoveMessage(id=m.id) for m in messages]})     

if "num" not in st.session_state:
    st.session_state.num = 0

@st.cache_resource(max_entries=1) # Why max_entries=1?
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


# Sidebar
model = update_api_and_model() # Run on_change then rerun!
authenticate()
max_token = get_max_tokens()
is_stream = stream_enabled()


chatbot = create_chatbot(st.session_state.api, st.session_state.api_key, model, max_token)

config = {"configurable": {"thread_id": "abc345"}}

system_prompt = get_system_prompt(on_change=clear_chat_history, args=(chatbot, config))

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
])

# Show chat History
if st.session_state["valid_auth"] and "messages" in chatbot.get_state(config).values:
    for msg in chatbot.get_state(config).values["messages"]:
        st.chat_message(msg.type).write(msg.content)

# Chat Interface
if query := st.chat_input(disabled=not st.session_state["valid_auth"]):
    st.chat_message(name="human").write(query)
    input_messages = [HumanMessage(query)]

    if not is_stream:
        output = chatbot.invoke({"messages": input_messages}, config)
        st.chat_message(name="assistant").write(output["messages"][-1].content)
    else:
        output = chatbot.stream({"messages": input_messages}, config, stream_mode="messages")
        st.chat_message(name="assistant").write(stream_wrapper(output))

