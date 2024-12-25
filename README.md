# 🤖 Chatbot

A Streamlit-based chatbot application that leverages LangChain framework to interact with multiple LLM providers.

## ✨ Features

- Support for multiple AI providers (OpenAI and Groq)
- Model selection for each provider:
    - OpenAI: gpt-4o-mini, gpt-4-turbo, gpt-4o
    - Groq: llama3-70b-8192, llama3-8b-8192
- Authentication via API key or password
- Configurable system prompts
- Message history with auto-trimming
- Token limit management

## ⚙️ Configuration

1. Create a [`.streamlit/secrets.toml`](.streamlit/secrets.toml) file:

```toml
PASSWORDS = ["your-password"]
OPENAI_API_KEY_DEV = "your-openai-key"
GROQ_API_KEY_DEV = "your-groq-key"
```

## 🚀 Installation

### Windows
```sh
pip install virtualenv
virtualenv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run Chatbot_LangChain.py
```

### Ubuntu/Linux
```sh
pip install virtualenv
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run Chatbot_LangChain.py
```

## 💡 Usage

1. Select your preferred API provider (OpenAI/Groq)
2. Choose a model
3. Set maximum token limit
4. Authenticate using API key or password
5. Optionally customize the system prompt
6. Start chatting!

## 📚 Resources

- [LangChain Chatbot Tutorial](https://python.langchain.com/docs/tutorials/chatbot/)
- [LangChain Message Trimming Guide](https://python.langchain.com/docs/how_to/trim_messages/)

## 📝 License

MIT License
