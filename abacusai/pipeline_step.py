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
            pythonFunction (PythonFunction): Information about the python function for the step.
    """

    def __init__(self, client, pipelineStepId=None, pipelineId=None, stepName=None, pipelineName=None, createdAt=None, updatedAt=None, pythonFunctionId=None, stepDependencies=None, cpuSize=None, memory=None, pythonFunction={}):
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
        self.python_function = client._build_class(
            PythonFunction, pythonFunction)

    def __repr__(self):
        return f"PipelineStep(pipeline_step_id={repr(self.pipeline_step_id)},\n  pipeline_id={repr(self.pipeline_id)},\n  step_name={repr(self.step_name)},\n  pipeline_name={repr(self.pipeline_name)},\n  created_at={repr(self.created_at)},\n  updated_at={repr(self.updated_at)},\n  python_function_id={repr(self.python_function_id)},\n  step_dependencies={repr(self.step_dependencies)},\n  cpu_size={repr(self.cpu_size)},\n  memory={repr(self.memory)},\n  python_function={repr(self.python_function)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'pipeline_step_id': self.pipeline_step_id, 'pipeline_id': self.pipeline_id, 'step_name': self.step_name, 'pipeline_name': self.pipeline_name, 'created_at': self.created_at, 'updated_at': self.updated_at, 'python_function_id': self.python_function_id, 'step_dependencies': self.step_dependencies, 'cpu_size': self.cpu_size, 'memory': self.memory, 'python_function': self._get_attribute_as_dict(self.python_function)}

    def delete(self):
        """
        Deletes a step from a pipeline.

        Args:
            pipeline_step_id (str): The ID of the pipeline step.
        """
        return self.client.delete_pipeline_step(self.pipeline_step_id)

    def update(self, function_name: str = None, source_code: str = None, step_input_mappings: list = None, output_variable_mappings: list = None, step_dependencies: list = None, package_requirements: list = None, cpu_size: str = None, memory: int = None):
        """
        Creates a step in a given pipeline.

        Args:
            function_name (str): The name of the Python function.
            source_code (str): Contents of a valid Python source code file. The source code should contain the transform feature group functions. A list of allowed imports and system libraries for each language is specified in the user functions documentation section.
            step_input_mappings (list): List of Python function arguments.
            output_variable_mappings (list): List of Python function ouputs.
            step_dependencies (list): List of step names this step depends on.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            cpu_size (str): Size of the CPU for the step function.
            memory (int): Memory (in GB) for the step function.

        Returns:
            PipelineStep: Object describing the pipeline.
        """
        return self.client.update_pipeline_step(self.pipeline_step_id, function_name, source_code, step_input_mappings, output_variable_mappings, step_dependencies, package_requirements, cpu_size, memory)

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
