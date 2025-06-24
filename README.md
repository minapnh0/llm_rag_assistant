# LLM RAG Assistant — GenAI Copilot Over Enterprise Documents

## Overview

This is a production-grade Retrieval-Augmented Generation (RAG) assistant built using:

- Hugging Face LLMs
- LangChain for chaining and prompt management
- FastAPI backend
- FAISS for vector-based document retrieval
- Config-based structure for local or cloud deployment

The goal is to simulate an internal GenAI copilot (for example, in an enterprise like Walmart) that can answer associate queries by retrieving relevant context from internal documents and generating accurate, grounded responses.

## Problem Statement

Large language models like ChatGPT or GPT-4 are not connected to your internal files (PDFs, textbooks, or company policies). As a result, they may generate incorrect or hallucinated answers because they do not have access to the real content.

This assistant solves that by:

- Reading and embedding your uploaded documents
- Identifying what type of question the user is asking (document-related or general)
- Returning answers grounded in those documents, or using a general LLM if appropriate

## Key Features

- Semantic search through uploaded documents using FAISS
- Query classification using BERT (to route between GPT and RAG)
- RAG pipeline that combines document retrieval and generation
- FastAPI backend with REST APIs
- Optional Streamlit frontend for simple UI testing

## How It Works

1. You upload PDFs or documents
2. The system embeds them using a vector store (FAISS)
3. When a question is asked:
   - It classifies if the query should use RAG (document-based) or GPT (general)
   - If document-based: it retrieves relevant content using vector similarity
   - It then generates a grounded response using the retrieved context

## What is BERT Used For

BERT is a type of model used to understand language. In this project, it is used for intent classification. It decides whether the user's query should be answered using document search (RAG) or a general model (GPT).

## What is RAG (Retrieval-Augmented Generation)

RAG combines search and generation:

- It retrieves relevant chunks from documents using vector similarity
- Then it uses a language model to generate a response based on those chunks

Example:
You ask: “What is PCA?”
The system finds pages from your document that mention PCA, and then generates an answer based on them.


## Local Setup Instructions

### 1. Clone the repository

git clone git@github.com:minapnh0/llm_rag_assistant.git
cd llm_rag_assistant

### 2. Create a virtual environment

python3 -m venv venv

### 3. Activate the virtual environment

On Mac/Linux:
source venv/bin/activate

On Windows:
.\venv\Scripts\activate

### 4. Install dependencies

pip install -r requirements.txt

### 5. Set up environment variables

Create a `.env` file in the project root:

HUGGINGFACE_API_TOKEN=your_token_here  
MODEL_ID=google/flan-t5-base  
EMBEDDING_MODEL=all-MiniLM-L6-v2

### 6. Add mock documents

Put your test PDFs in:

data/mock_docs/

## Running the Application

### Run the FastAPI backend (Development mode)

uvicorn app.main:app --reload

This starts the server at http://127.0.0.1:8000  
API documentation is available at http://127.0.0.1:8000/docs

### Run the Streamlit frontend (optional UI)

streamlit run ui/app_ui.py

### Run in Production Mode (Gunicorn)

(Optional for deployment)

1. Install Gunicorn:

pip install gunicorn

2. Run with multiple workers:

gunicorn -w 4 -b 0.0.0.0:8080 --preload app.main:app

## Developer Notes

- FAISS index creation and embeddings are handled in `app/embedder/`
- RAG orchestration logic is in `app/rag/rag_service.py`
- Prompt formatting and templates can be adjusted in `app/prompt_templates.py`
- Request logging and traceability are handled in `app/utils/logger_utils.py`
