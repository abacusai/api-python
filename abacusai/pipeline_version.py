from .api_class import PythonFunctionArgument
from .code_source import CodeSource
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
            stepVersions (PipelineStepVersion): A list of the pipeline step versions.
            codeSource (CodeSource): information on the source code
            pipelineVariableMappings (PythonFunctionArgument): A description of the function variables into the pipeline.
    """

    def __init__(self, client, pipelineName=None, pipelineId=None, pipelineVersion=None, createdAt=None, updatedAt=None, completedAt=None, status=None, error=None, stepVersions={}, codeSource={}, pipelineVariableMappings={}):
        super().__init__(client, pipelineVersion)
        self.pipeline_name = pipelineName
        self.pipeline_id = pipelineId
        self.pipeline_version = pipelineVersion
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.completed_at = completedAt
        self.status = status
        self.error = error
        self.step_versions = client._build_class(
            PipelineStepVersion, stepVersions)
        self.code_source = client._build_class(CodeSource, codeSource)
        self.pipeline_variable_mappings = client._build_class(
            PythonFunctionArgument, pipelineVariableMappings)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'pipeline_name': repr(self.pipeline_name), f'pipeline_id': repr(self.pipeline_id), f'pipeline_version': repr(self.pipeline_version), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at), f'completed_at': repr(
            self.completed_at), f'status': repr(self.status), f'error': repr(self.error), f'step_versions': repr(self.step_versions), f'code_source': repr(self.code_source), f'pipeline_variable_mappings': repr(self.pipeline_variable_mappings)}
        class_name = "PipelineVersion"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'pipeline_name': self.pipeline_name, 'pipeline_id': self.pipeline_id, 'pipeline_version': self.pipeline_version, 'created_at': self.created_at, 'updated_at': self.updated_at, 'completed_at': self.completed_at, 'status': self.status,
                'error': self.error, 'step_versions': self._get_attribute_as_dict(self.step_versions), 'code_source': self._get_attribute_as_dict(self.code_source), 'pipeline_variable_mappings': self._get_attribute_as_dict(self.pipeline_variable_mappings)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

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

    def reset(self, steps: list = None, include_downstream_steps: bool = True):
        """
        Reruns a pipeline version for the given steps and downstream steps if specified.

        Args:
            steps (list): List of pipeline step names to rerun.
            include_downstream_steps (bool): Whether to rerun downstream steps from the steps you have passed

        Returns:
            PipelineVersion: Object describing the pipeline version
        """
        return self.client.reset_pipeline_version(self.pipeline_version, steps, include_downstream_steps)

    def list_logs(self):
        """
        Gets the logs for the steps in a given pipeline version.

        Args:
            pipeline_version (str): The id of the pipeline version.

        Returns:
            PipelineVersionLogs: Object describing the logs for the steps in the pipeline.
        """
        return self.client.list_pipeline_version_logs(self.pipeline_version)

    def skip_pending_steps(self):
        """
        Skips pending steps in a pipeline version.

        Args:
            pipeline_version (str): The id of the pipeline version.

        Returns:
            PipelineVersion: Object describing the pipeline version
        """
        return self.client.skip_pending_pipeline_version_steps(self.pipeline_version)

    def wait_for_pipeline(self, timeout=1200):
        """
        A waiting call until all the stages in a pipeline version have completed.

        Args:
            timeout (int): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'RUNNING'}, timeout=timeout)

    def get_status(self):
        """
        Gets the status of the pipeline version.

        Returns:
            str: A string describing the status of a pipeline version (pending, running, complete, etc.).
        """
        return self.describe().status
