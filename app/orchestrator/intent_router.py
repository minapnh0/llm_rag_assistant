"""
intent_router.py

Routes user queries to either GPT or RAG pipeline based on classified intent.
"""

from app.classification.intent_classifier import BERTIntentClassifier
from app.generation.gpt_generator import GPTService
from app.rag.rag_service import RAGService
from app.utils.logger_utils import get_logger

logger = get_logger("LLMOrchestrator")


class LLMOrchestrator:
    def __init__(self, config: dict = None):
        self.config = config or {}

        try:
            self.bert = BERTIntentClassifier()
            logger.info("Intent classifier initialized.")
        except Exception as e:
            logger.exception("Failed to load BERTIntentClassifier.")
            raise e

        try:
            self.gpt = GPTService()
            logger.info("GPT service initialized.")
        except Exception as e:
            logger.exception("Failed to initialize GPTService.")
            raise e

        try:
            self.rag = RAGService.from_config(self.config.get("rag", {}))
            logger.info("RAG service initialized.")
        except Exception as e:
            logger.exception("Failed to initialize RAGService.")
            raise e

    def handle_query(self, query: str, trace_id: str = None) -> dict:
        try:
            # Force intent for now; replace with actual classifier if needed
            intent = "doc_question"
            logger.info(f"[DEBUG] Using intent: {intent}", extra={"trace_id": trace_id})

            if intent == "doc_question":
                rag_result = self.rag.query(query)

                response_text = rag_result.get("result")
                source_docs = rag_result.get("source_documents", [])
                error = rag_result.get("error")

                if not response_text:
                    return {
                        "response": "Sorry, I couldn't find anything relevant in the documents.",
                        "intent": intent,
                        "source_docs": [],
                        "error": error,
                    }

                return {
                    "response": response_text,
                    "intent": intent,
                    "source_docs": [doc.get("source", "unknown") for doc in source_docs],
                    "error": error,
                }

            else:
                logger.info("Routing query to GPT generator.")
                response_text = self.gpt.generate_response(query, trace_id=trace_id)
                return {
                    "response": response_text,
                    "intent": intent,
                    "source_docs": None,
                    "error": None,
                }

        except Exception as e:
            logger.exception("Query handling pipeline failed.")
            return {
                "response": None,
                "intent": "error",
                "source_docs": None,
                "error": str(e),
            }
