import streamlit as st

# Define Shared Session State Between Pages
if "valid_auth" not in st.session_state:
    st.session_state.valid_auth = False

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "password_auth" not in st.session_state:
    st.session_state.password_auth = False

if "api" not in st.session_state:
    st.session_state.api = ""

pg = st.navigation({
    "Chatbots": [
        st.Page("modules/0_Chatbot_LangChain.py", title="Chatbot Langchain", icon=":material/chat:")
    ],
    "Summarization": [
        st.Page("modules/1_Summarization_Stuff.py", title="Stuff Technique", icon=":material/summarize:"),
        st.Page("modules/2_Summarization_Map_Reduce.py", title="Map-Reduce Technique", icon=":material/summarize:")
    ],
    "RAG": [
        st.Page("modules/3_RAG.py", title="PDF Q/A", icon=":material/docs:")
    ]
})

pg.run()
