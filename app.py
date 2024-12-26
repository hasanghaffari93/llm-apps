import streamlit as st



# Each page is a script executed from your entrypoint file. \
# You can define a page from a Python file or function. 
# If you include elements or widgets in your entrypoint file, 
# they become common elements between your pages.

# TODO: Find good icon from https://fonts.google.com/icons?icon.set=Material+Symbols&icon.style=Rounded

pg = st.navigation({
    "Chatbots": [
        st.Page("modules/0_Chatbot_LangChain.py", title="Chatbot Langchain", icon=":material/dashboard:")
    ],
    "Summarization": [
        st.Page("modules/1_Summarization_Stuff.py", title="Stuff Technique", icon=":material/dashboard:"),
        st.Page("modules/2_Summarization_Map_Reduce.py", title="Map-Reduce Technique", icon=":material/dashboard:")
    ],
    "RAG": [
        st.Page("modules/3_RAG.py", title="PDF Q/A", icon=":material/dashboard:")
    ]
})

pg.run()
