import logging

class LLMService:
    def __init__(self):
        logging.info(f"{self.__class__.__name__} initialized.")

    def call_api(self, prompt: str, system_prompt: str = None) -> str:
        """
        Makes an API call and returns the response.
        Should be implemented by subclasses.
        """
        logging.error("call_api() not implemented.")
        raise NotImplementedError("call_api() must be implemented in subclass.")

    def count_tokens(self, prompt: str, system_prompt: str = None) -> int:
        """
        Counts the number of tokens in a prompt.
        Should be implemented by subclasses.
        """
        logging.error("count_tokens() not implemented.")
        raise NotImplementedError("count_tokens() must be implemented in subclass.")

    def _int_service(self, api_key):
        """
        Initializes the service connection. Should be implemented by subclasses.
        """
        logging.error("_int_service() not implemented.")
        raise NotImplementedError("_int_service() must be implemented in subclass.")

    def _get_servive(self):
        """
        Returns the service client. Should be implemented by subclasses.
        """
        logging.error("_get_servive() not implemented.")
        raise NotImplementedError("_get_servive() must be implemented in subclass.")

    def _test_service_connection(self):
        """
        Tests the service connection. Should be implemented by subclasses.
        """
        logging.error("_test_service_connection() not implemented.")
        raise NotImplementedError(
            "_test_service_connection() must be implemented in subclass."
        )
