# LLM RAG Assistant — GenAI Copilot Over Enterprise Documents

## Overview

This project is a **production-grade Retrieval-Augmented Generation (RAG) assistant** built using:

- Hugging Face LLMs
- LangChain for chaining & prompt management
- FastAPI backend
- FAISS for vector-based document retrieval
- Config-ready structure for cloud deployment

The goal is to simulate an **internal GenAI copilot** (e.g., for Walmart) that can answer associate queries by retrieving accurate context from internal policy documents and generating concise, trustworthy responses.

---

The problem:
Typical LLMs (like ChatGPT or GPT-4) are not connected to your personal PDFs or textboo

ks. They may give wrong answers or hallucinate because they don’t have access to the real content.

This app solves that by:

Reading and understanding your uploaded books using a search system.
Detecting what kind of question you're asking (e.g., general or document-based).
Giving accurate answers by combining a smart search (RAG) with a language model (LLM).

## What is BERT (Used for Intent Detection)
BERT is a type of neural network that understands the meaning of a sentence.

In this project, we use BERT to classify the type of your question:
Is it about the uploaded book? → Route to the document system (RAG).
Is it a general question like “What is AI?” → Use a general LLM (GPT).

## What is RAG ?
RAG = Search + Generate

It combines:
Search through your uploaded books using vector similarity (like how Google finds pages).
Generate an answer using a language model (like GPT) with the search results.

Example:
You ask: “What is PCA?”
The system searches for PCA in the book.
It gives those pages to GPT to write a clear answer, based only on your book.
-------------------------------------
## Project Structure


-------------------------------------
## Local Setup Instructions

### Project Bootstrap
1. Clone the repo (or navigate to project root):
   cd ~/Desktop/llm_rag_assistant

2. Create a virtual environment:
   python3 -m venv venv

3. Activate the virtual environment:
   - On Mac/Linux:
     source venv/bin/activate
   - On Windows:
     .\venv\Scripts\activate

4. Install all dependencies:
   pip install -r requirements.txt

5. Create .env file in the project root:
   Add the following entries:
   HUGGINGFACE_API_TOKEN=your_token_here
   MODEL_ID=google/flan-t5-base
   EMBEDDING_MODEL=all-MiniLM-L6-v2

6. Add mock documents to:
   data/mock_docs/

-------------------------------------

## Running the Application

## Run backend
### Run with Uvicorn (Dev Mode):
   uvicorn app.main:app --reload
   uvicorn app.main:app --host 127.0.0.1 --port 8000

# In another terminal, run UI
streamlit run ui/app_ui.py

- Server will start at: http://127.0.0.1:8000
- API documentation auto-generated at: http://127.0.0.1:8000/docs

-------------------------------------

### Run with Gunicorn (Production Mode)

(Optional for deployment testing)

1. Install Gunicorn:
   pip install gunicorn

2. Run with 4 workers:
   gunicorn -w 4 -b 0.0.0.0:8080 --preload app.main:app

-------------------------------------

## Developer Notes

- Embeddings will be managed via app/embedder.py using FAISS
- RAG flow is built in app/rag_engine.py
- Prompt templates can be versioned in app/prompt_templates.py
- Logging and tracing hooks can be added under logs/

pip install -r requirements.txt
# llm_rag_assistant
