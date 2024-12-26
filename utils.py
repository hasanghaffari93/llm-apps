import streamlit as st
from typing import Optional, Callable
import openai

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


def select_api_and_model(
        api_provider: Optional[tuple[str, ...]]=None, 
        on_change: Optional[Callable]=None
    ) -> None:

    with st.sidebar.expander("Select API and Model"):
        st.selectbox(
            label="API provider",
            options=AppConfig.API_PROVIDERS if api_provider is None else api_provider,
            index=0,
            key="api",
            on_change=on_change,
        )

        st.selectbox(
            label="model name",
            options=AppConfig.MODEL_NAMES[st.session_state["api"]],
            index=0,
            key="model",
            on_change=on_change,
        )

def authenticate():
    with st.sidebar.popover("Enter API Key or Password", use_container_width=True):
        auth_type = st.radio(
            "Authenticate via",
            ["Your API Key", "Password"],
            key="auth_type",
            horizontal=True,
            label_visibility="collapsed",
            captions=["Use your own API key", "Use a password provided by the author for testing"],
        )

        label = f"{st.session_state['api']} API Key:" if auth_type == "Your API Key" else "Password:"
        
        auth_input = st.text_input(
            label=label,
            key="auth",
            type="password",
        )

        st.session_state["valid_auth"] = False
        st.session_state["api_key"] = ""

        if auth_type == "Your API Key":
            if validate_api_key(auth_input, st.session_state["api"]):
                st.session_state["api_key"] = auth_input
                st.session_state["valid_auth"] = True
                st.sidebar.success("Valid API key")
            else:
                st.sidebar.error("Invalid API key")
        else:
            if auth_input in st.secrets["PASSWORDS"]:
                st.session_state["api_key"] = st.secrets[AppConfig.KEY_NAMES[st.session_state["api"]]]
                st.session_state["valid_auth"] = True
                st.sidebar.success("Valid password")
            else:
                st.sidebar.error("Invalid password")


@st.cache_data()
def validate_api_key(api_key: str, api: str) -> bool:
    try:
        client = openai.OpenAI(
            base_url=AppConfig.BASE_URLS[api],
            api_key=api_key,
        )
        client.models.list()
        return True
    except openai.AuthenticationError:
        return False
    except Exception as error:
        st.sidebar.error(f"API error: {error}")
        return False
    

def get_max_tokens(on_change: Optional[Callable]=None) -> None:
    st.sidebar.number_input(
        label="Trim messages if tokens exceed:",
        min_value=10,
        max_value=128000,
        value=AppConfig.DEFAULT_MAX_TOKENS,
        step=1000,
        key="max_tokens",
        help="Long context costs a lot!",
        placeholder="3000",
        on_change=on_change,
)

def get_context_length_limit(on_change: Optional[Callable]=None) -> None:
    st.sidebar.number_input(
        label="Context length limit:",
        min_value=10,
        max_value=128000,
        value=AppConfig.DEFAULT_MAX_TOKENS,
        step=1000,
        key="context_length_limit", 
        help="Long context costs a lot!",
        placeholder="3000",
        on_change=on_change,
    )

def stream_enabled():
    st.sidebar.checkbox(
        label="Enable stream chat",
        value=True,
        key="stream",
        help="The output will be streaming",
    )

def get_system_prompt(on_change: Optional[Callable]=None) -> None:
    st.text_input(
        label="System Prompt:",
        value=AppConfig.DEFAULT_SYSTEM_PROMPT,
        max_chars=1000,
        key="system_prompt",
        help="Top-level instructions for the model's behavior",
        placeholder="System Prompt",
        on_change=on_change,
    )

def select_embedding_model(on_change: Optional[Callable]=None) -> None:
    with st.sidebar.expander("Select Embedding Eodel"):
        st.selectbox(
            label="OpenAI Embedding Model",
            options=AppConfig.EMBEDDING_MODEL_NAMES["OpenAI"],
            index=0,
            key="embedding_model",
            on_change=on_change,
        )
    