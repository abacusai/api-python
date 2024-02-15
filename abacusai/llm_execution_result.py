from .execute_feature_group_operation import ExecuteFeatureGroupOperation
from .llm_execution_preview import LlmExecutionPreview
from .return_class import AbstractApiClass


class LlmExecutionResult(AbstractApiClass):
    """
        Results of executing queries using LLM.

        Args:
            client (ApiClient): An authenticated API Client instance
            status (str): The status of the execution.
            error (str): The error message if the execution failed.
            execution (ExecuteFeatureGroupOperation): Information on execution of the query.
            preview (LlmExecutionPreview): Preview of executing queries using LLM.
    """

    def __init__(self, client, status=None, error=None, execution={}, preview={}):
        super().__init__(client, None)
        self.status = status
        self.error = error
        self.execution = client._build_class(
            ExecuteFeatureGroupOperation, execution)
        self.preview = client._build_class(LlmExecutionPreview, preview)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'status': repr(self.status), f'error': repr(
            self.error), f'execution': repr(self.execution), f'preview': repr(self.preview)}
        class_name = "LlmExecutionResult"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'status': self.status, 'error': self.error, 'execution': self._get_attribute_as_dict(
            self.execution), 'preview': self._get_attribute_as_dict(self.preview)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
