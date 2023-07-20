from .pipeline_step_version_reference import PipelineStepVersionReference
from .return_class import AbstractApiClass


class PipelineStepVersion(AbstractApiClass):
    """
        A version of a pipeline step.

        Args:
            client (ApiClient): An authenticated API Client instance
            stepName (str): The name of the step.
            pipelineStepVersion (str): The reference to the pipeline step version.
            pipelineStepId (str): The reference to this step.
            pipelineId (str): The reference to the pipeline this step belongs to.
            pipelineVersion (str): The reference to the pipeline version.
            createdAt (str): The date and time which this step was created.
            updatedAt (str): The date and time when this step was last updated.
            status (str): The status of the pipeline version.
            error (str): The error message if the pipeline step failed.
            outputErrors (str): The error message of a pipeline step's output.
            pythonFunctionId (str): The reference to the python function
            functionVariableMappings (dict): The mappings for function parameters' names.
            stepDependencies (list[str]): List of steps this step depends on.
            outputVariableMappings (dict): The mappings for the output variables to the step.
            cpuSize (str): CPU size specified for the step function.
            memory (int): Memory in GB specified for the step function.
            pipelineStepVersionReferences (PipelineStepVersionReference): A list to the output instances of the pipeline step version.
    """

    def __init__(self, client, stepName=None, pipelineStepVersion=None, pipelineStepId=None, pipelineId=None, pipelineVersion=None, createdAt=None, updatedAt=None, status=None, error=None, outputErrors=None, pythonFunctionId=None, functionVariableMappings=None, stepDependencies=None, outputVariableMappings=None, cpuSize=None, memory=None, pipelineStepVersionReferences={}):
        super().__init__(client, pipelineStepVersion)
        self.step_name = stepName
        self.pipeline_step_version = pipelineStepVersion
        self.pipeline_step_id = pipelineStepId
        self.pipeline_id = pipelineId
        self.pipeline_version = pipelineVersion
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.status = status
        self.error = error
        self.output_errors = outputErrors
        self.python_function_id = pythonFunctionId
        self.function_variable_mappings = functionVariableMappings
        self.step_dependencies = stepDependencies
        self.output_variable_mappings = outputVariableMappings
        self.cpu_size = cpuSize
        self.memory = memory
        self.pipeline_step_version_references = client._build_class(
            PipelineStepVersionReference, pipelineStepVersionReferences)

    def __repr__(self):
        return f"PipelineStepVersion(step_name={repr(self.step_name)},\n  pipeline_step_version={repr(self.pipeline_step_version)},\n  pipeline_step_id={repr(self.pipeline_step_id)},\n  pipeline_id={repr(self.pipeline_id)},\n  pipeline_version={repr(self.pipeline_version)},\n  created_at={repr(self.created_at)},\n  updated_at={repr(self.updated_at)},\n  status={repr(self.status)},\n  error={repr(self.error)},\n  output_errors={repr(self.output_errors)},\n  python_function_id={repr(self.python_function_id)},\n  function_variable_mappings={repr(self.function_variable_mappings)},\n  step_dependencies={repr(self.step_dependencies)},\n  output_variable_mappings={repr(self.output_variable_mappings)},\n  cpu_size={repr(self.cpu_size)},\n  memory={repr(self.memory)},\n  pipeline_step_version_references={repr(self.pipeline_step_version_references)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'step_name': self.step_name, 'pipeline_step_version': self.pipeline_step_version, 'pipeline_step_id': self.pipeline_step_id, 'pipeline_id': self.pipeline_id, 'pipeline_version': self.pipeline_version, 'created_at': self.created_at, 'updated_at': self.updated_at, 'status': self.status, 'error': self.error, 'output_errors': self.output_errors, 'python_function_id': self.python_function_id, 'function_variable_mappings': self.function_variable_mappings, 'step_dependencies': self.step_dependencies, 'output_variable_mappings': self.output_variable_mappings, 'cpu_size': self.cpu_size, 'memory': self.memory, 'pipeline_step_version_references': self._get_attribute_as_dict(self.pipeline_step_version_references)}

    def get_step_version_logs(self):
        """
        Gets the logs for a given step version.

        Args:
            pipeline_step_version (str): The id of the pipeline step version.

        Returns:
            PipelineStepVersionLogs: Object describing the pipeline step logs.
        """
        return self.client.get_step_version_logs(self.pipeline_step_version)
