from .code_source import CodeSource
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
            codeSource (CodeSource): Information about the source code of the pipeline step version.
    """

    def __init__(self, client, stepName=None, pipelineStepVersion=None, pipelineStepId=None, pipelineId=None, pipelineVersion=None, createdAt=None, updatedAt=None, status=None, error=None, outputErrors=None, pythonFunctionId=None, functionVariableMappings=None, stepDependencies=None, outputVariableMappings=None, cpuSize=None, memory=None, pipelineStepVersionReferences={}, codeSource={}):
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
        self.code_source = client._build_class(CodeSource, codeSource)

    def __repr__(self):
        repr_dict = {f'step_name': repr(self.step_name), f'pipeline_step_version': repr(self.pipeline_step_version), f'pipeline_step_id': repr(self.pipeline_step_id), f'pipeline_id': repr(self.pipeline_id), f'pipeline_version': repr(self.pipeline_version), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at), f'status': repr(self.status), f'error': repr(self.error), f'output_errors': repr(
            self.output_errors), f'python_function_id': repr(self.python_function_id), f'function_variable_mappings': repr(self.function_variable_mappings), f'step_dependencies': repr(self.step_dependencies), f'output_variable_mappings': repr(self.output_variable_mappings), f'cpu_size': repr(self.cpu_size), f'memory': repr(self.memory), f'pipeline_step_version_references': repr(self.pipeline_step_version_references), f'code_source': repr(self.code_source)}
        class_name = "PipelineStepVersion"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'step_name': self.step_name, 'pipeline_step_version': self.pipeline_step_version, 'pipeline_step_id': self.pipeline_step_id, 'pipeline_id': self.pipeline_id, 'pipeline_version': self.pipeline_version, 'created_at': self.created_at, 'updated_at': self.updated_at, 'status': self.status, 'error': self.error, 'output_errors': self.output_errors, 'python_function_id': self.python_function_id,
                'function_variable_mappings': self.function_variable_mappings, 'step_dependencies': self.step_dependencies, 'output_variable_mappings': self.output_variable_mappings, 'cpu_size': self.cpu_size, 'memory': self.memory, 'pipeline_step_version_references': self._get_attribute_as_dict(self.pipeline_step_version_references), 'code_source': self._get_attribute_as_dict(self.code_source)}
        return {key: value for key, value in resp.items() if value is not None}

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            PipelineStepVersion: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Describes a pipeline step version.

        Args:
            pipeline_step_version (str): The ID of the pipeline step version.

        Returns:
            PipelineStepVersion: An object describing the pipeline step version.
        """
        return self.client.describe_pipeline_step_version(self.pipeline_step_version)

    def get_step_version_logs(self):
        """
        Gets the logs for a given step version.

        Args:
            pipeline_step_version (str): The id of the pipeline step version.

        Returns:
            PipelineStepVersionLogs: Object describing the pipeline step logs.
        """
        return self.client.get_step_version_logs(self.pipeline_step_version)
