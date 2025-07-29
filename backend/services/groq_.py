from backend.services.general import LLMService
from groq import Groq
import os
import logging
import textwrap
from dotenv import load_dotenv

load_dotenv()

DEFAULT_GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

class GroqService(LLMService):
    def __init__(self, model=DEFAULT_GROQ_MODEL, api_key=None):
        self.model = model
        if api_key:
            self._int_service(api_key=api_key)
        elif os.getenv("GROQ_API_KEY"):
            self._int_service(api_key=os.getenv("GROQ_API_KEY"))
        else:
            logging.error("GROQ_API_KEY environment variable not set")
            raise ValueError("GROQ_API_KEY environment variable not set")

        super().__init__()

    def _int_service(self, api_key):
        self.client = Groq(api_key=api_key)
        logging.info("Initialized Groq client.")

    def call_api(self, prompt: str, system_prompt=None) -> str:
        # Replace with actual Groq API call
        if not self.client:
            raise RuntimeError("Groq client not available")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt or "You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content

        if content is None:
            logging.warning(f"Groq API returned None content")
            return ""

        result = content.strip()
        logging.debug(f"Groq API response: {result[:100]}...")

        return result

    def count_tokens(self, prompt: str) -> int:
        return len(prompt.split())

    def _get_servive(self):
        return self.client
