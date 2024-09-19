from .llm_code_block import LlmCodeBlock
from .return_class import AbstractApiClass


class LlmResponse(AbstractApiClass):
    """
        The response returned by LLM

        Args:
            client (ApiClient): An authenticated API Client instance
            content (str): Full response from LLM.
            tokens (int): The number of tokens in the response.
            stopReason (str): The reason due to which the response generation stopped.
            llmName (str): The name of the LLM model used to generate the response.
            inputTokens (int): The number of input tokens used in the LLM call.
            outputTokens (int): The number of output tokens generated in the LLM response.
            totalTokens (int): The total number of tokens (input + output) used in the LLM interaction.
            codeBlocks (LlmCodeBlock): A list of parsed code blocks from raw LLM Response
    """

    def __init__(self, client, content=None, tokens=None, stopReason=None, llmName=None, inputTokens=None, outputTokens=None, totalTokens=None, codeBlocks={}):
        super().__init__(client, None)
        self.content = content
        self.tokens = tokens
        self.stop_reason = stopReason
        self.llm_name = llmName
        self.input_tokens = inputTokens
        self.output_tokens = outputTokens
        self.total_tokens = totalTokens
        self.code_blocks = client._build_class(LlmCodeBlock, codeBlocks)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'content': repr(self.content), f'tokens': repr(self.tokens), f'stop_reason': repr(self.stop_reason), f'llm_name': repr(self.llm_name), f'input_tokens': repr(
            self.input_tokens), f'output_tokens': repr(self.output_tokens), f'total_tokens': repr(self.total_tokens), f'code_blocks': repr(self.code_blocks)}
        class_name = "LlmResponse"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'content': self.content, 'tokens': self.tokens, 'stop_reason': self.stop_reason, 'llm_name': self.llm_name, 'input_tokens': self.input_tokens,
                'output_tokens': self.output_tokens, 'total_tokens': self.total_tokens, 'code_blocks': self._get_attribute_as_dict(self.code_blocks)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
