import streamlit as st
from openai import OpenAI

# Constants
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

def restart_chat():
    if "system_prompt" in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": st.session_state["system_prompt"]}
        ]

st.title("🤖 Chatbot with OpenAI Library")
st.caption("🚀 Let's chat with different models using OpenAI's client libraries")

# Layout
col1, col2 = st.columns(2)

with col1:
    api_provider = st.selectbox(
        label="Which API do you want to use?",
        options=API_PROVIDERS,
        index=0,
        key="api_provider",
        help="This app has API key of these providers",
        placeholder="Select an API...",
        on_change=restart_chat,
    )

    if api_provider:
        model_name = st.selectbox(
            label="Which model do you want to use?",
            options=MODEL_NAMES[api_provider],
            index=0,
            key="model_name",
            help="This app has API key of these model",
            placeholder="Select a model...",
            on_change=restart_chat,
        )

with col2:
    st.text_input(
        label="System Prompt:",
        value="You are a helpful assistant.",
        max_chars=100,
        key="system_prompt",
        help="Top-level instructions for the model's behavior",
        placeholder="System Prompt",
        on_change=restart_chat,
    )

    stream_enabled = st.checkbox(
        label="Enable stream chat",
        value=True,
        key="stream",
        help="The output will be streaming",
    )

# Initialize OpenAI client
client = OpenAI(
    base_url=BASE_URLS[api_provider],
    api_key=st.secrets[KEY_NAMES[api_provider]],
)

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": st.session_state["system_prompt"]}
    ]

# Display system prompt
st.chat_message(name='assistant', avatar="🤖").write(
    st.session_state["system_prompt"]
)

# Display chat history
for msg in st.session_state.messages[1:]:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle new messages
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message(name="user").write(prompt)

    if stream_enabled:
        stream = client.chat.completions.create(
            model=model_name,
            messages=st.session_state.messages,
            stream=True
        )
        msg = st.chat_message("assistant").write_stream(stream)
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=st.session_state.messages
        )
        msg = response.choices[0].message.content
        st.chat_message(name="assistant").write(msg)

    st.session_state.messages.append({"role": "assistant", "content": msg})

