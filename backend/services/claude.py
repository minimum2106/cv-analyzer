from general import LLMService
import anthropic
import logging
import os

DEFAULT_CLAUDE_MODEL = "claude-2"


class ClaudeService(LLMService):
    def __init__(self, api_key=None, model=DEFAULT_CLAUDE_MODEL):
        if api_key:
            self._int_service(api_key=api_key)
        elif os.getenv("CLAUDE_API_KEY"):
            self._int_service(api_key=os.getenv("CLAUDE_API_KEY"))
        else:
            logging.error("CLAUDE_API_KEY environment variable not set")
            raise ValueError("CLAUDE_API_KEY environment variable not set")

        self.model = model
        super().__init__()

    def _int_service(self, api_key):
        # Initialize Claude client here
        self.client = anthropic.Anthropic(api_key)  # Replace with actual client
        logging.info("Initialized Claude client.")

    def call_api(self, system_prompt: str, prompt: dict) -> str:
        logging.info("Calling Claude API (mock).")
        # Replace with actual Claude API call
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=1,
            system=system_prompt,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )

        logging.debug(f"Claude response: {response.content[:100]}...")

        return response.content

    def count_tokens(self, prompt: str, system_prompt: str = None) -> int:
        logging.info("Counting tokens with Claude (mock).")
        response = self.client.messages.count_tokens(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        num_tokens = response.json().get("input_tokens", 0)
        logging.info(f"Counted {num_tokens} tokens for prompt.")
        return num_tokens

    def _get_servive(self):
        return self.client

    def _test_service_connection(self):
        logging.info("Testing Claude service connection (mock).")
        return True  # Implement actual test
