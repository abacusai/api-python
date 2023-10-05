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
            featureGroupOperation (ExecuteFeatureGroupOperation): Operation for executing queries on feature group.
            preview (LlmExecutionPreview): Preview of executing queries using LLM.
    """

    def __init__(self, client, status=None, error=None, featureGroupOperation={}, preview={}):
        super().__init__(client, None)
        self.status = status
        self.error = error
        self.feature_group_operation = client._build_class(
            ExecuteFeatureGroupOperation, featureGroupOperation)
        self.preview = client._build_class(LlmExecutionPreview, preview)

    def __repr__(self):
        return f"LlmExecutionResult(status={repr(self.status)},\n  error={repr(self.error)},\n  feature_group_operation={repr(self.feature_group_operation)},\n  preview={repr(self.preview)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'status': self.status, 'error': self.error, 'feature_group_operation': self._get_attribute_as_dict(self.feature_group_operation), 'preview': self._get_attribute_as_dict(self.preview)}
