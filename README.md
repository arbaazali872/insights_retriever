# InsightVault

A local-first RAG (Retrieval-Augmented Generation) chatbot for document analysis. Upload PDFs or articles and query them using natural language - entirely offline with no API costs.

## Features

- **Document Processing**: Support for URLs, PDFs, and text files
- **Conversational Memory**: Context-aware responses across chat sessions
- **Local Execution**: Runs entirely on your machine using Ollama
- **Source Citations**: Answers include references to source documents
- **Async Operations**: Non-blocking UI with streaming responses

## Tech Stack

- **UI**: Chainlit
- **LLM**: Ollama (llama3.2:3b)
- **Vector Store**: ChromaDB
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Framework**: LangChain

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed and running
- 16GB RAM recommended

## Installation

1. Clone the repository:
```bash
git clone https://github.com/arbaazali872/insights_retriever.git
cd insights_retriever
```

2. Create and activate virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Pull the Ollama model:
```bash
ollama pull llama3.2:3b
```

## Usage

1. Start the application:
```bash
chainlit run app.py
```

2. Open your browser at `http://localhost:8000`

3. Add documents:
   - Use `/add <url>` to load articles
   - Upload PDF or text files directly

4. Ask questions about your documents

## Commands

- `/add <url>` - Add article from URL
- `/stats` - View knowledge base statistics
- `/clear` - Clear conversation history
- `/reset` - Clear all documents and history

## Project Structure

```
insight-vault/
├── app.py                    # Main application entry point
├── requirements.txt          # Python dependencies
├── config/
│   ├── settings.py          # Configuration settings
│   └── prompts.py           # LLM prompts
├── core/
│   ├── document_loader.py   # Document processing
│   ├── vectorstore.py       # ChromaDB operations
│   ├── llm_manager.py       # Ollama LLM wrapper
│   └── rag_chain.py         # RAG pipeline
└── data/
    ├── chroma_db/           # Vector database storage
    └── documents/           # Uploaded files
```

## Configuration

Edit `config/settings.py` to customize:

- `OLLAMA_MODEL`: Change LLM model
- `CHUNK_SIZE`: Adjust document chunking
- `TOP_K_RESULTS`: Number of retrieved documents
- `TEMPERATURE`: LLM response randomness

## Limitations

- No authentication (single-user)
- In-memory session storage
- Limited to supported document formats
- Requires Ollama running locally

## License

MIT

## Acknowledgments

Built with [Chainlit](https://chainlit.io), [LangChain](https://langchain.com), and [Ollama](https://ollama.ai).