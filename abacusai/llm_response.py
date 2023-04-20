from .llm_code_block import LlmCodeBlock
from .return_class import AbstractApiClass


class LlmResponse(AbstractApiClass):
    """
        The response returned by LLM

        Args:
            client (ApiClient): An authenticated API Client instance
            content (str): Content of the response.
            tokens (int): The number of tokens in the response.
            stopReason (str): The reason due to which the response generation stopped.
            llmCodeBlock (LlmCodeBlock): Parsed code block from raw LLM Response
    """

    def __init__(self, client, content=None, tokens=None, stopReason=None, llmCodeBlock={}):
        super().__init__(client, None)
        self.content = content
        self.tokens = tokens
        self.stop_reason = stopReason
        self.llm_code_block = client._build_class(LlmCodeBlock, llmCodeBlock)

    def __repr__(self):
        return f"LlmResponse(content={repr(self.content)},\n  tokens={repr(self.tokens)},\n  stop_reason={repr(self.stop_reason)},\n  llm_code_block={repr(self.llm_code_block)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'content': self.content, 'tokens': self.tokens, 'stop_reason': self.stop_reason, 'llm_code_block': self._get_attribute_as_dict(self.llm_code_block)}
