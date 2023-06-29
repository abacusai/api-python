from .return_class import AbstractApiClass


class PipelineStepVersionLogs(AbstractApiClass):
    """
        Logs for a given pipeline step version.

        Args:
            client (ApiClient): An authenticated API Client instance
            stepName (str): The name of the step
            pipelineStepId (str): The ID of the step
            pipelineStepVersion (str): The version of the step
            logs (str): The logs for both stdout and stderr of the step
    """

    def __init__(self, client, stepName=None, pipelineStepId=None, pipelineStepVersion=None, logs=None):
        super().__init__(client, None)
        self.step_name = stepName
        self.pipeline_step_id = pipelineStepId
        self.pipeline_step_version = pipelineStepVersion
        self.logs = logs

    def __repr__(self):
        return f"PipelineStepVersionLogs(step_name={repr(self.step_name)},\n  pipeline_step_id={repr(self.pipeline_step_id)},\n  pipeline_step_version={repr(self.pipeline_step_version)},\n  logs={repr(self.logs)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'step_name': self.step_name, 'pipeline_step_id': self.pipeline_step_id, 'pipeline_step_version': self.pipeline_step_version, 'logs': self.logs}
