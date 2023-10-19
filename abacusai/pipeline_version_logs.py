from .pipeline_step_version_logs import PipelineStepVersionLogs
from .return_class import AbstractApiClass


class PipelineVersionLogs(AbstractApiClass):
    """
        Logs for a given pipeline version.

        Args:
            client (ApiClient): An authenticated API Client instance
            stepLogs (PipelineStepVersionLogs): A list of the pipeline step version logs.
    """

    def __init__(self, client, stepLogs={}):
        super().__init__(client, None)
        self.step_logs = client._build_class(PipelineStepVersionLogs, stepLogs)

    def __repr__(self):
        repr_dict = {f'step_logs': repr(self.step_logs)}
        class_name = "PipelineVersionLogs"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'step_logs': self._get_attribute_as_dict(self.step_logs)}
        return {key: value for key, value in resp.items() if value is not None}
