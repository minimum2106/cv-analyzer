from general import LLMService
from openai import OpenAI
import tiktoken
import logging
import os

DEFAULT_OPENAI_MODEL = "gpt-4o"

class OpenAIService(LLMService):
    def __init__(self, api_key=None, model=DEFAULT_OPENAI_MODEL):
        if api_key:
            self._int_service(api_key=api_key)
        elif os.getenv("OPENAI_API_KEY"):
            self._int_service(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            logging.error("OPENAI_API_KEY environment variable not set")
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        super().__init__()
        self.model = model

    def _int_service(self, api_key):
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key = api_key
        )

        logging.info("Initialized OpenAI client.")

    def call_api(self, prompt: str, system_prompt: str = None) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=None,
            input=prompt,
        )

        result = response.output_text.strip()
        logging.debug(f"OpenAI response: {result[:100]}...")
        return result

    def count_tokens(self, prompt: str) -> int:
        encoding = tiktoken.encoding_for_model(self.model)
        num_tokens = len(encoding.encode(prompt))
        logging.info(f"Counted {num_tokens} tokens for prompt.")
        return num_tokens

    def _get_servive(self):
        return self.client
