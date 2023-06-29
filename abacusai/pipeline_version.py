from .pipeline_step_version import PipelineStepVersion
from .return_class import AbstractApiClass


class PipelineVersion(AbstractApiClass):
    """
        A version of a pipeline.

        Args:
            client (ApiClient): An authenticated API Client instance
            pipelineName (str): The name of the pipeline this step is a part of.
            pipelineId (str): The reference to the pipeline this step belongs to.
            pipelineVersion (str): The reference to this pipeline version.
            createdAt (str): The date and time which this pipeline version was created.
            updatedAt (str): The date and time which this pipeline version was updated.
            completedAt (str): The date and time which this pipeline version was updated.
            status (str): The status of the pipeline version.
            error (str): The relevant error, if the status is FAILED.
            pipelineVariableMappings (dict): A description of the function variables into the pipeline.
            stepVersions (PipelineStepVersion): A list of the pipeline step versions.
    """

    def __init__(self, client, pipelineName=None, pipelineId=None, pipelineVersion=None, createdAt=None, updatedAt=None, completedAt=None, status=None, error=None, pipelineVariableMappings=None, stepVersions={}):
        super().__init__(client, pipelineVersion)
        self.pipeline_name = pipelineName
        self.pipeline_id = pipelineId
        self.pipeline_version = pipelineVersion
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.completed_at = completedAt
        self.status = status
        self.error = error
        self.pipeline_variable_mappings = pipelineVariableMappings
        self.step_versions = client._build_class(
            PipelineStepVersion, stepVersions)

    def __repr__(self):
        return f"PipelineVersion(pipeline_name={repr(self.pipeline_name)},\n  pipeline_id={repr(self.pipeline_id)},\n  pipeline_version={repr(self.pipeline_version)},\n  created_at={repr(self.created_at)},\n  updated_at={repr(self.updated_at)},\n  completed_at={repr(self.completed_at)},\n  status={repr(self.status)},\n  error={repr(self.error)},\n  pipeline_variable_mappings={repr(self.pipeline_variable_mappings)},\n  step_versions={repr(self.step_versions)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'pipeline_name': self.pipeline_name, 'pipeline_id': self.pipeline_id, 'pipeline_version': self.pipeline_version, 'created_at': self.created_at, 'updated_at': self.updated_at, 'completed_at': self.completed_at, 'status': self.status, 'error': self.error, 'pipeline_variable_mappings': self.pipeline_variable_mappings, 'step_versions': self._get_attribute_as_dict(self.step_versions)}

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            PipelineVersion: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Describes a specified pipeline version

        Args:
            pipeline_version (str): Unique string identifier for the pipeline version

        Returns:
            PipelineVersion: Object describing the pipeline version
        """
        return self.client.describe_pipeline_version(self.pipeline_version)

    def list_logs(self):
        """
        Gets the logs for the steps in a given pipeline version.

        Args:
            pipeline_version (str): The id of the pipeline version.

        Returns:
            PipelineVersionLogs: Object describing the logs for the steps in the pipeline.
        """
        return self.client.list_pipeline_version_logs(self.pipeline_version)

    def wait_for_pipeline(self, timeout=1200):
        """
        A waiting call until all the stages in a pipeline version have completed.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'RUNNING'}, timeout=timeout)

    def get_status(self):
        """
        Gets the status of the pipeline version.

        Returns:
            str: A string describing the status of a pipeline version (pending, running, complete, etc.).
        """
        return self.describe().status
