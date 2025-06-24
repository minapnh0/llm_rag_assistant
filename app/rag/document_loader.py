"""
document_loader.py

Loads and splits PDF documents from a specified folder into text chunks
for embedding and vector search using LangChain.
"""

import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.config.settings import get_settings
from app.utils.logger_utils import get_logger

logger = get_logger("DocumentLoader")
settings = get_settings()


def load_and_split_documents(pdf_folder: str = None) -> List[Document]:
    """
    Loads all PDF documents from a folder, extracts their text, attaches metadata,
    and splits them into chunks using RecursiveCharacterTextSplitter.

    Args:pdf_folder (str, optional): Folder path containing PDFs.
            If not provided, falls back to settings.DOCS_PATH.
    Returns:List[Document]: List of LangChain Document chunks ready for embedding.
    """
    folder_path = pdf_folder or settings.DOCS_PATH
    docs = []

    if not os.path.exists(folder_path):
        logger.error(f"Folder does not exist: {folder_path}")
        return []

    logger.info(f"Scanning folder: {folder_path}")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            logger.info(f"Loading file: {file_path}")

            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()

                for page_num, page in enumerate(pages):
                    page.metadata["filename"] = filename
                    page.metadata["page_number"] = page_num + 1
                    docs.append(page)

            except Exception as e:
                logger.exception(f"Failed to load PDF: {filename}", extra={"error": str(e)})

    if not docs:
        logger.warning("No PDF documents were loaded.")
        return []

    logger.info(f"Loaded {len(docs)} pages from {len(os.listdir(folder_path))} PDF files.")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(docs)

    logger.info(f"Split into {len(chunks)} total chunks.")
    return chunks
