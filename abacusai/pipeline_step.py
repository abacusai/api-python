from typing import List

from .code_source import CodeSource
from .python_function import PythonFunction
from .return_class import AbstractApiClass


class PipelineStep(AbstractApiClass):
    """
        A step in a pipeline.

        Args:
            client (ApiClient): An authenticated API Client instance
            pipelineStepId (str): The reference to this step.
            pipelineId (str): The reference to the pipeline this step belongs to.
            stepName (str): The name of the step.
            pipelineName (str): The name of the pipeline this step is a part of.
            createdAt (str): The date and time which this step was created.
            updatedAt (str): The date and time when this step was last updated.
            pythonFunctionId (str): The python function_id.
            stepDependencies (list[str]): List of steps this step depends on.
            cpuSize (str): CPU size specified for the step function.
            memory (int): Memory in GB specified for the step function.
            timeout (int): Timeout for the step in minutes, default is 300 minutes.
            pythonFunction (PythonFunction): Information about the python function for the step.
            codeSource (CodeSource): Information about the source code of the step function.
    """

    def __init__(self, client, pipelineStepId=None, pipelineId=None, stepName=None, pipelineName=None, createdAt=None, updatedAt=None, pythonFunctionId=None, stepDependencies=None, cpuSize=None, memory=None, timeout=None, pythonFunction={}, codeSource={}):
        super().__init__(client, pipelineStepId)
        self.pipeline_step_id = pipelineStepId
        self.pipeline_id = pipelineId
        self.step_name = stepName
        self.pipeline_name = pipelineName
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.python_function_id = pythonFunctionId
        self.step_dependencies = stepDependencies
        self.cpu_size = cpuSize
        self.memory = memory
        self.timeout = timeout
        self.python_function = client._build_class(
            PythonFunction, pythonFunction)
        self.code_source = client._build_class(CodeSource, codeSource)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'pipeline_step_id': repr(self.pipeline_step_id), f'pipeline_id': repr(self.pipeline_id), f'step_name': repr(self.step_name), f'pipeline_name': repr(self.pipeline_name), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at), f'python_function_id': repr(
            self.python_function_id), f'step_dependencies': repr(self.step_dependencies), f'cpu_size': repr(self.cpu_size), f'memory': repr(self.memory), f'timeout': repr(self.timeout), f'python_function': repr(self.python_function), f'code_source': repr(self.code_source)}
        class_name = "PipelineStep"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'pipeline_step_id': self.pipeline_step_id, 'pipeline_id': self.pipeline_id, 'step_name': self.step_name, 'pipeline_name': self.pipeline_name, 'created_at': self.created_at, 'updated_at': self.updated_at, 'python_function_id': self.python_function_id,
                'step_dependencies': self.step_dependencies, 'cpu_size': self.cpu_size, 'memory': self.memory, 'timeout': self.timeout, 'python_function': self._get_attribute_as_dict(self.python_function), 'code_source': self._get_attribute_as_dict(self.code_source)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def delete(self):
        """
        Deletes a step from a pipeline.

        Args:
            pipeline_step_id (str): The ID of the pipeline step.
        """
        return self.client.delete_pipeline_step(self.pipeline_step_id)

    def update(self, function_name: str = None, source_code: str = None, step_input_mappings: List = None, output_variable_mappings: List = None, step_dependencies: list = None, package_requirements: list = None, cpu_size: str = None, memory: int = None, timeout: int = None):
        """
        Creates a step in a given pipeline.

        Args:
            function_name (str): The name of the Python function.
            source_code (str): Contents of a valid Python source code file. The source code should contain the transform feature group functions. A list of allowed imports and system libraries for each language is specified in the user functions documentation section.
            step_input_mappings (List): List of Python function arguments.
            output_variable_mappings (List): List of Python function outputs.
            step_dependencies (list): List of step names this step depends on.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            cpu_size (str): Size of the CPU for the step function.
            memory (int): Memory (in GB) for the step function.
            timeout (int): Timeout for the pipeline step, default is 300 minutes.

        Returns:
            PipelineStep: Object describing the pipeline.
        """
        return self.client.update_pipeline_step(self.pipeline_step_id, function_name, source_code, step_input_mappings, output_variable_mappings, step_dependencies, package_requirements, cpu_size, memory, timeout)

    def rename(self, step_name: str):
        """
        Renames a step in a given pipeline.

        Args:
            step_name (str): The name of the step.

        Returns:
            PipelineStep: Object describing the pipeline.
        """
        return self.client.rename_pipeline_step(self.pipeline_step_id, step_name)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            PipelineStep: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Deletes a step from a pipeline.

        Args:
            pipeline_step_id (str): The ID of the pipeline step.

        Returns:
            PipelineStep: An object describing the pipeline step.
        """
        return self.client.describe_pipeline_step(self.pipeline_step_id)
