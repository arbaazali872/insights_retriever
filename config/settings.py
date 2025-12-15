import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
DOCS_DIR = DATA_DIR / "documents"

# Create directories
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# LLM Settings
OLLAMA_MODEL = "llama3.2:3b"  # Lightweight for 16GB RAM
OLLAMA_BASE_URL = "http://localhost:11434"

# Embedding Settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

# Chunking Settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retrieval Settings
TOP_K_RESULTS = 5
TEMPERATURE = 0.1

# Memory Settings
MAX_CONVERSATION_HISTORY = 5