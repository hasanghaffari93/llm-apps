# LLM App Collection

This project is a collection of applications leveraging Large Language Models (LLMs) for various tasks such as chatbots, document summarization, and retrieval-augmented generation (RAG). The applications are built using LangChain and LangGraph for the backend processing and Streamlit for the UI.


## Features
### 🤖 Chatbot
- Support for multiple LLM providers (`OpenAI` and `Groq`)
- Model selection for each provider:
    - `OpenAI`: `gpt-4o-mini`, `gpt-4-turbo`, `gpt-4o`
    - `Groq`: `llama3-70b-8192`, `llama3-8b-8192`
- Message history with auto-trimming

### 📝 Document Summarization
- Two summarization techniques:
  - `Stuff`: For shorter documents that fit within context window
  - `Map-Reduce`: For longer documents that exceed context limits
- PDF document support

### 🔍 RAG (Retrieval Augmented Generation)
- PDF document question-answering
- `FAISS` vector store for efficient similarity search
- `OpenAI` embeddings integration

### ⚙️ Technical Features
- Built with `Streamlit` for interactive UI
- `LangChain` and `LangGraph` integration for LLM operations
- Modular architecture with separate pages
- Authentication via API key or password


## Demo App 

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-apps.streamlit.app/)
s

## Setup

1. Clone the repository:

```sh
git clone https://github.com/yourusername/llm-app-suite.git
cd llm-app-suite
```

2. Configure API keys in [`.streamlit/secrets.toml`](.streamlit/secrets.toml) file:

```toml
OPENAI_API_KEY_DEV = "your-openai-key"
GROQ_API_KEY_DEV = "your-groq-key"
PASSWORDS = ["your-password"]
```

3. Create and activate virtual environment:

Ubuntu/Linux
```sh
pip install virtualenv
virtualenv .venv
source .venv/bin/activate
```

Windows:
```sh
pip install virtualenv
virtualenv venv
venv\Scripts\activate.bat
```

4. Install dependencies:
```sh
pip install -r requirements.txt
```

5. Run the application:

```sh
streamlit run app.py
```