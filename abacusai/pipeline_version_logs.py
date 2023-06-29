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
        return f"PipelineVersionLogs(step_logs={repr(self.step_logs)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'step_logs': self._get_attribute_as_dict(self.step_logs)}
