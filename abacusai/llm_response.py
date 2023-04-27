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
            codeBlocks (LlmCodeBlock): A list of parsed code blocks from raw LLM Response
    """

    def __init__(self, client, content=None, tokens=None, stopReason=None, codeBlocks={}):
        super().__init__(client, None)
        self.content = content
        self.tokens = tokens
        self.stop_reason = stopReason
        self.code_blocks = client._build_class(LlmCodeBlock, codeBlocks)

    def __repr__(self):
        return f"LlmResponse(content={repr(self.content)},\n  tokens={repr(self.tokens)},\n  stop_reason={repr(self.stop_reason)},\n  code_blocks={repr(self.code_blocks)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'content': self.content, 'tokens': self.tokens, 'stop_reason': self.stop_reason, 'code_blocks': self._get_attribute_as_dict(self.code_blocks)}
