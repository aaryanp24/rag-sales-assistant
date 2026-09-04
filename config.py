"""
Central config. All keys are read from environment variables (or a local .env file).
Everything here has a free tier:

- GROQ_API_KEY    -> https://console.groq.com/keys           (free, fast Llama models)
- GEMINI_API_KEY  -> https://aistudio.google.com/apikey       (free tier, used as fallback)
- TAVILY_API_KEY  -> https://app.tavily.com                   (free tier, 1000 searches/mo)

You don't need all three. LLM_PROVIDER picks which one app.py calls.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM provider ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "gemini"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# --- Web research ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --- Embeddings (local, free, no key needed) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Storage paths ---
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "data")

# --- Chunking ---
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # characters of overlap between chunks
