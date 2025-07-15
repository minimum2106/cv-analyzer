class LLMService:
    def __init__(self):
        self.default_client = None
        self.api_key = None

    def call_api(self, system_prompt: str, data: dict) -> str:
        """
        Makes an API call returns the response.
        """
        
        pass
    

    def count_tokens(self, prompt: str) -> int:
        """
        Counts the number of tokens in a prompt.
        """

        pass
    
    def _int_service(self):
        """
        Initializes the service connection. This method should be implemented
        """

        pass

    def _get_servive(self):
        """
        Returns the service client. This method should be implemented
        """

        pass

    def _test_service_connection(self):
        """
        Tests the service connection. This method should be implemented
        """

        pass

