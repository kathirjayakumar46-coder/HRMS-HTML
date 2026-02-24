import os

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
RAG_K = int(os.getenv("RAG_K", "5"))