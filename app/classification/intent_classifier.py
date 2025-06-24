
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from typing import List, Union
from functools import lru_cache

from app.utils.logger_utils import get_logger

logger = get_logger("BERTIntentClassifier")


@lru_cache(maxsize=1)
def get_model_and_tokenizer(model_name: str):
    """
    Lazily loads and caches the tokenizer and model to prevent repeated loading.
    Args: model_name (str): Name of the HuggingFace model (e.g., "bert-base-uncased")
    Returns: Tuple[tokenizer, model]: HuggingFace tokenizer and PyTorch model
    """
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name)
        logger.info(f"Loaded model: {model_name}")
        return tokenizer, model
    except Exception as e:
        logger.exception(f"Error loading model or tokenizer: {e}")
        raise RuntimeError("Failed to load BERT model or tokenizer")


class BERTIntentClassifier:
    """
    Intent classifier using BERT and HuggingFace Transformers.
    Attributes:
        model_name (str): The model identifier from HuggingFace
        label_map (dict): Mapping from prediction index to human-readable label
        device (torch.device): Automatically set to 'cuda' if available, else 'cpu'
    """

    def __init__(self, model_name: str = "bert-base-uncased", label_map: dict = None):
        """
        Initializes the classifier with model and tokenizer.

        Args:
            model_name (str): Pretrained BERT model name from HuggingFace hub.
            label_map (dict, optional): Mapping of class indices to labels. Defaults to {0: "doc_question", 1: "general"}.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer, self.model = get_model_and_tokenizer(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.label_map = label_map or {
            0: "rag",
            1: "gpt"
        }

    def classify(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Classify one or more input texts into intent categories.
        Args:texts (str or List[str]): Input text(s) to classify.
        Returns: str or List[str]: Predicted label(s) corresponding to each input.
                              Returns a single label if input is a single string.
        Raises: RuntimeError: If inference or tokenization fails.
        """
        try:
            if isinstance(texts, str):
                texts = [texts]

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                predictions = torch.argmax(logits, dim=1).tolist()

            results = [self.label_map.get(pred, "unknown") for pred in predictions]
            logger.info(f"Classified input(s): {texts} => {results}")
            return results[0] if len(results) == 1 else results

        except Exception as e:
            logger.exception(f"Failed to classify input(s): {texts} - {e}")
            raise RuntimeError("Classification failed")
