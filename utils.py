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
    DEFAULT_MAX_TOKENS = 128000
    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

def reset_auth():
    if st.session_state.valid_auth:
        if not st.session_state.password_auth:
            st.session_state.valid_auth = False
        else:
            st.session_state.api_key = st.secrets[AppConfig.KEY_NAMES[st.session_state.api]]

def update_api_and_model(from_api=None):

    with st.sidebar.expander("Select API and Model"):
        st.session_state.api = st.selectbox(
            label="API provider",
            options=AppConfig.API_PROVIDERS if from_api is None else from_api,
            index=0,
            on_change=reset_auth(),
        )

        model = st.selectbox(
            label="model name",
            options=AppConfig.MODEL_NAMES[st.session_state.api],
            index=0,
            # key="model",
            # on_change=on_change,
        )

    return model

def authenticate():
    
    with st.sidebar.expander("Enter API Key or Password", expanded=not st.session_state.valid_auth):

        auth_type = st.radio(
            "Authenticate via",
            ["Your API Key", "Password"],
            # key="auth_type",
            index=1,
            horizontal=True,
            label_visibility="collapsed",
            captions=["Use your own API key", "Use a password provided by the author for testing"],
            disabled=st.session_state.valid_auth
        )

        label = f"{st.session_state.api} API Key:" if auth_type == "Your API Key" else "Password:"
        
        auth_input = st.text_input(
            label=label,
            type="password",
            disabled=st.session_state.valid_auth
        )

        button = st.button('Submit', disabled=st.session_state.valid_auth)

        if button and not st.session_state.valid_auth:
            
            if auth_type == "Your API Key":
                if validate_api_key(auth_input, st.session_state.api):
                    st.session_state.api_key = auth_input
                    st.session_state.valid_auth = True
                    st.session_state.password_auth= False
                    st.rerun()
                else:
                    st.sidebar.error("Invalid API key")
            else:
                if auth_input in st.secrets["PASSWORDS"]:
                    st.session_state.api_key = st.secrets[AppConfig.KEY_NAMES[st.session_state.api]]
                    st.session_state.valid_auth = True
                    st.session_state.password_auth= True
                    st.rerun()
                else:
                    st.sidebar.error("Invalid password")

    if st.session_state.valid_auth:
        st.sidebar.success("Valid authentication")
    else:
        st.error("Validate your authentication")                    


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
    max_tokens = st.sidebar.number_input(
        label="Trim messages if tokens exceed:",
        min_value=10,
        max_value=128000,
        value=AppConfig.DEFAULT_MAX_TOKENS,
        step=1000,
        # key="max_tokens",
        help="Long context costs a lot!",
        placeholder="128000",
        on_change=on_change,
    )
    return max_tokens

def get_context_length_limit(on_change: Optional[Callable]=None) -> None:
    context_length_limit = st.sidebar.number_input(
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
    return context_length_limit

def stream_enabled():
    stream = st.sidebar.checkbox(
        label="Enable stream chat",
        value=True,
        help="The output will be streaming",
    )
    return stream

def get_system_prompt(on_change: Optional[Callable]=None, args=None) -> None:
    system_prompt = st.text_input(
        label="System Prompt:",
        value=AppConfig.DEFAULT_SYSTEM_PROMPT,
        max_chars=1000,
        help="Top-level instructions for the model's behavior",
        placeholder="System Prompt",
        on_change=on_change,
        args=args
    )
    return system_prompt

def select_embedding_model(on_change: Optional[Callable]=None) -> None:
    with st.sidebar.expander("Select Embedding Model"):
        st.selectbox(
            label="OpenAI Embedding Model",
            options=AppConfig.EMBEDDING_MODEL_NAMES["OpenAI"],
            index=0,
            key="embedding_model",
            on_change=on_change,
        )
    