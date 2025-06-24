import re
import time
from typing import Optional, Dict, List, Union

from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI

from app.config.settings import get_settings
from app.utils.logger_utils import get_logger

logger = get_logger("RAGService")
settings = get_settings()


class RAGService:
    def __init__(
        self,
        index_path: str = settings.FAISS_INDEX_PATH,
        embedding_model: str = settings.EMBED_MODEL,
        model_name: str = settings.MODEL_NAME,  # "gpt-3.5-turbo"
        k: int = settings.TOP_K,
        llm: Optional[ChatOpenAI] = None
    ):
        """
        Initialize RAGService and its chain using FAISS + OpenAI.

        Args:
            index_path (str): Path to FAISS index.
            embedding_model (str): Embedding model (HuggingFace).
            model_name (str): OpenAI model like gpt-3.5-turbo.
            k (int): Top-K retrieval.
            llm (Optional): Injected LLM.
        """
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.k = k
        self.llm = llm
        self.chain = self._build_rag_chain()

    @classmethod
    def from_config(cls, config: Dict):
        return cls(
            index_path=config.get("index_path", settings.FAISS_INDEX_PATH),
            embedding_model=config.get("embedding_model", settings.EMBED_MODEL),
            model_name=config.get("model_name", settings.MODEL_NAME),
            k=config.get("top_k", settings.TOP_K),
        )

    def _build_rag_chain(self) -> RetrievalQA:
        try:
            logger.info("Initializing RAG chain...")
            logger.info(f"Embedding model: {self.embedding_model}")
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

            logger.info(f"Loading FAISS index from: {self.index_path}")
            db = FAISS.load_local(
                self.index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )

            if not self.llm:
                logger.info(f"Loading OpenAI model: {self.model_name}")
                self.llm = ChatOpenAI(
                    model_name=self.model_name,
                    temperature=0,
                    openai_api_key=settings.OPENAI_API_KEY
                )

            retriever = db.as_retriever(search_kwargs={"k": self.k})

            chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )

            logger.info("RAG chain ready.")
            return chain

        except Exception as e:
            logger.exception("Failed to initialize RAG chain.")
            raise RuntimeError("RAG chain initialization failed") from e

    def query(self, question: str) -> Dict[str, Optional[Union[str, List[Dict]]]]:
        try:
            logger.info(f"RAG received question: {question}")
            start_time = time.time()

            response = self.chain({"query": question})
            logger.info(f"Raw chain response: {response}")

            retrieved_docs = self.chain.retriever.get_relevant_documents(question)
            logger.info(f"Top FAISS chunks: {[doc.page_content[:200] for doc in retrieved_docs]}")

            result_text = self._clean_text(response.get("result", ""))
            source_docs = self._format_source_documents(response.get("source_documents", []))

            return {
                "result": result_text,
                "source_documents": source_docs,
                "error": None
            }

        except Exception as e:
            logger.exception("RAG query failed.")
            return {
                "result": None,
                "source_documents": [],
                "error": str(e)
            }

    def _format_source_documents(self, docs: List[Document]) -> List[Dict[str, str]]:
        return [
            {
                "source": doc.metadata.get("filename", "unknown"),
                "page": doc.metadata.get("page_number", "n/a"),
                "snippet": self._clean_text(doc.page_content[:300])
            }
            for doc in docs
        ]

    def _clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.strip())
