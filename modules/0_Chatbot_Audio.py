from typing import Annotated, Sequence, Dict, Literal
from typing_extensions import TypedDict
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, BaseMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages, RemoveMessage
from utils import get_max_tokens, get_system_prompt, update_api_and_model, authenticate, stream_enabled
from openai import OpenAI

if "retry" not in st.session_state:
    st.session_state["retry"] = False

def clear_chat_history(chatbot, config):
    if chatbot and chatbot.get_state(config).values:
        messages = chatbot.get_state(config).values["messages"]
        chatbot.update_state(config, {"messages": [RemoveMessage(id=m.id) for m in messages]})

def show_chat_history(chatbot, config):
    if st.session_state.get("valid_auth") and "messages" in chatbot.get_state(config).values:
        messages = chatbot.get_state(config).values["messages"]
        if st.session_state.retry:
            messages = messages[:-1]
        for msg in messages:
            st.chat_message(msg.type).write(msg.content)        

@st.cache_resource(max_entries=1) # Why max_entries=1?
def create_chatbot(api: str, api_key: str, model: str, max_tokens: int, system_prompt: str):

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

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])    

    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        retry: bool

    workflow = StateGraph(state_schema=State)

    def should_retry(state: State) -> Literal["remove_last_ai_message", "call_model"]:
        if "retry" in state and state["retry"]:
            return "remove_last_ai_message"
        else:
            return "call_model"
        
    def remove_last_ai_message(state: State) -> State:
        messages = state["messages"]
        if len(messages) > 1:
            if messages[-1].type == "ai":
                return {"messages": [RemoveMessage(id=messages[-1].id)], "retry": False}
            else:
                raise ValueError("Last message is not an AI message")


    def call_model(state: State) -> Dict[str, list]:
        trimmed_messages = trimmer.invoke(state["messages"])
        prompt = prompt_template.invoke({"messages": trimmed_messages})
        response = llm.invoke(prompt)
        return {"messages": [response]}
    
    workflow.add_node("remove_last_ai_message", remove_last_ai_message)
    workflow.add_node("call_model", call_model)

    workflow.add_conditional_edges(START, should_retry)
    workflow.add_edge("remove_last_ai_message", "call_model")
    workflow.add_edge("call_model", END)

    checkpointer = MemorySaver()
    
    return workflow.compile(checkpointer=checkpointer)


def stream_wrapper(stream):
    for chunk, _ in stream:
        yield chunk.content   

def write_ai_message(state, chatbot, config, is_stream):
    with st.chat_message(name="assistant"):
        if not is_stream:
            output = chatbot.invoke(state, config)
            st.write(output["messages"][-1].content)
        else:
            stream = chatbot.stream(state, config, stream_mode="messages")
            output = st.write_stream(stream_wrapper(stream))

            with st.spinner('Running Text to Speech'):
                client = OpenAI(api_key=st.session_state.api_key)
                response = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    response_format="wav",
                    input=output,
                )
                st.audio(response.read(), autoplay=True)            

# UI Setup
st.header(":material/chat: Chatbot")
st.caption("🚀 Powered by LangChain | 🔥 OpenAI LLMs, Whisper, and TTS | 🛠️ Streamlit UI")


model = update_api_and_model(from_api="OpenAI")
authenticate()
max_token = get_max_tokens()
is_stream = stream_enabled()


config = {"configurable": {"thread_id": "abc345"}}

system_prompt = get_system_prompt()

chatbot = create_chatbot(st.session_state.api, st.session_state.api_key, model, max_token, system_prompt)

show_chat_history(chatbot, config)

if st.session_state.retry:
    st.session_state["retry"] = False
    write_ai_message({"retry": True}, chatbot, config, is_stream)

audio = st._bottom.audio_input("Record a voice message", disabled=not st.session_state["valid_auth"])

if audio:
    client = OpenAI(api_key=st.session_state.api_key)
    with st.spinner('Transcribing audio...'):
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            response_format="text"
        )
    query = transcript
    st.chat_message(name="human").write(query)
    write_ai_message({"messages": [HumanMessage(query)]}, chatbot, config, is_stream)

if st.session_state["valid_auth"] and \
    "messages" in chatbot.get_state(config).values and \
    chatbot.get_state(config).values["messages"] and \
    chatbot.get_state(config).values["messages"][-1].type == "ai":
    if st.button("Retry"):
        st.session_state["retry"] = True
        st.rerun()

if st.sidebar.button("New Chat", disabled=not st.session_state["valid_auth"]):
    clear_chat_history(chatbot, config)
    st.rerun()
