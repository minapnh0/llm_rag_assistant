"""
Embeds PDF documents into vector representations using a HuggingFace model,
then stores the vectors in a FAISS index locally for future retrieval.
"""

import os
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.rag.document_loader import load_and_split_documents
from app.config.settings import get_settings
from app.utils.logger_utils import get_logger

logger = get_logger("Embedder")
settings = get_settings()


def embed_and_store(
    pdf_folder: Optional[str] = None,
    index_path: Optional[str] = None,
    embedding_model: Optional[str] = None
) -> Optional[str]:
    """
    Loads and embeds documents from a folder and stores the vector index using FAISS.

    Args:
        pdf_folder (str, optional): Path to PDF folder. Defaults to settings.DOCS_PATH.
        index_path (str, optional): Path to save FAISS index. Defaults to settings.FAISS_INDEX_PATH.
        embedding_model (str, optional): HuggingFace model to use. Defaults to settings.EMBED_MODEL.

    Returns:
        Optional[str]: Path to saved FAISS index, or None if failure occurred.
    """
    try:
        pdf_folder = pdf_folder or settings.DOCS_PATH
        index_path = index_path or settings.FAISS_INDEX_PATH
        embedding_model = embedding_model or settings.EMBED_MODEL

        if not os.path.isdir(pdf_folder):
            logger.error(f"PDF folder not found: {pdf_folder}")
            return None

        logger.info(f"Loading documents from: {pdf_folder}")
        docs = load_and_split_documents(pdf_folder)

        if not docs:
            logger.warning("No documents found to embed.")
            return None

        logger.info(f"Total document chunks: {len(docs)}")
        for i, doc in enumerate(docs[:3]):
            logger.debug(f"[Sample Chunk {i+1}] {doc.page_content[:200]}")

        logger.info(f"Using embedding model: {embedding_model}")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        logger.info("Creating FAISS vector index...")
        vector_store = FAISS.from_documents(docs, embeddings)

        os.makedirs(index_path, exist_ok=True)
        logger.info(f"Saving FAISS index to: {index_path}")
        vector_store.save_local(index_path)

        logger.info(f"FAISS index saved at: {index_path}")
        return index_path

    except Exception as e:
        logger.exception("Embedding and indexing failed.", extra={"error": str(e)})
        return None
