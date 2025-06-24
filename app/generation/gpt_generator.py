from typing import Optional
from openai import OpenAI
from openai._exceptions import OpenAIError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config.settings import get_settings
from app.utils.logger_utils import get_logger

logger = get_logger("GPTService")


class GPTService:
    def __init__(self, model: Optional[str] = None, max_retries: int = 3, timeout: int = 15):
        settings = get_settings()

        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment or config.")

        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model or settings.MODEL_NAME
        self.max_retries = max_retries
        self.timeout = timeout

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=1, max=10),
        retry=retry_if_exception_type(RateLimitError)
    )
    def _call_openai(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        trace_id: Optional[str]
    ) -> str:
        logger.info("Calling OpenAI API with retry", extra={"trace_id": trace_id, "model": self.model})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout
        )
        return response.choices[0].message.content.strip()

    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        trace_id: Optional[str] = None
    ) -> Optional[str]:
        try:
            return self._call_openai(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                trace_id=trace_id
            )
        except RateLimitError:
            logger.warning("OpenAI rate limit exceeded even after retries.", extra={"trace_id": trace_id})
            return f"[MOCK] GPT response for prompt: {prompt}"

        except OpenAIError as oe:
            logger.exception("OpenAI API error", extra={"trace_id": trace_id, "error": str(oe)})
            return "OpenAI error occurred. Please try again later."

        except Exception as e:
            logger.exception("Unexpected GPTService error", extra={"trace_id": trace_id, "error": str(e)})
            return "Unexpected error occurred while generating GPT response."
