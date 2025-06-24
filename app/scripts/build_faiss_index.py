# scripts/build_faiss_index.py

import sys
import os
sys.path.append(os.path.abspath("app"))

from config.env_loader import load_env
from rag.embedder import embed_and_store

load_env()

if __name__ == "__main__":
    path = embed_and_store()
    if path:
        print(f"FAISS index saved at: {path}")
    else:
        print("Failed to build index.")
