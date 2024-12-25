from typing import Annotated, Sequence, Dict
from typing_extensions import TypedDict

import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, BaseMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages, RemoveMessage

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


def restart_chat() -> None:
    if chatbot and chatbot.get_state(config).values:
        messages = chatbot.get_state(config).values["messages"]
        chatbot.update_state(config, {"messages": [RemoveMessage(id=m.id) for m in messages]})


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


@st.cache_resource
def create_chatbot(api_provider: str, auth: str, max_tokens: int):

    if not auth:
        return None

    llm = ChatOpenAI(model=model_name, api_key=api_key, max_tokens=1000) if api_provider == "OpenAI" \
        else ChatGroq(model=model_name, api_key=api_key, max_tokens=1000)

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


# UI Setup
st.header("🤖 Chatbot with Langchain Framework")
st.caption("🚀 Let's chat with different models using Langchain")

# Sidebar Configuration
api_provider = st.sidebar.selectbox(
    label="Which API do you want to use?",
    options=AppConfig.API_PROVIDERS,
    index=0,
    key="api_provider",
    placeholder="Select an API...",
    on_change=restart_chat,
)

model_name = st.sidebar.selectbox(
    label="Which model do you want to use?",
    options=AppConfig.MODEL_NAMES[api_provider],
    index=0,
    key="model_name",
    placeholder="Select a model...",
    on_change=restart_chat,
)

max_tokens = st.sidebar.number_input(
    label="Trim messages if tokens exceed:",
    min_value=10,
    max_value=128000,
    value=AppConfig.DEFAULT_MAX_TOKENS,
    step=1000,
    key="max_tokens", 
    help="Long context costs a lot!",
    placeholder="3000",
    on_change=restart_chat,
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


# Layout
st.text_input(
    label="System Prompt:",
    value=AppConfig.DEFAULT_SYSTEM_PROMPT,
    max_chars=1000,
    key="system_prompt",
    help="Top-level instructions for the model's behavior",
    placeholder="System Prompt",
    on_change=restart_chat,
)


prompt_template = ChatPromptTemplate.from_messages([
    ("system", st.session_state["system_prompt"]),
    MessagesPlaceholder(variable_name="messages"),
])

chatbot = create_chatbot(api_provider, auth, max_tokens)
config = {"configurable": {"thread_id": "abc345"}}

# Chat Interface

# Show chat History
if chatbot and "messages" in chatbot.get_state(config).values:
    for msg in chatbot.get_state(config).values["messages"]:
        st.chat_message(msg.type).write(msg.content)

if query := st.chat_input(disabled=not st.session_state["valid_auth"]):
    st.chat_message(name="human").write(query)
    input_messages = [HumanMessage(query)]

    if not stream_enabled:
        output = chatbot.invoke({"messages": input_messages}, config)
        st.chat_message(name="assistant").write(output["messages"][-1].content)
    else:
        stream = chatbot.stream({"messages": input_messages}, config, stream_mode="messages")
        with st.empty():
            output = ""
            for chunk, _ in stream:
                output += chunk.content
                st.chat_message(name="assistant").write(output)
