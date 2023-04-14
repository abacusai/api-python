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
            pythonFunction (PythonFunction): Information about the python function for the step.
    """

    def __init__(self, client, pipelineStepId=None, pipelineId=None, stepName=None, pipelineName=None, createdAt=None, updatedAt=None, pythonFunctionId=None, stepDependencies=None, pythonFunction={}):
        super().__init__(client, pipelineStepId)
        self.pipeline_step_id = pipelineStepId
        self.pipeline_id = pipelineId
        self.step_name = stepName
        self.pipeline_name = pipelineName
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.python_function_id = pythonFunctionId
        self.step_dependencies = stepDependencies
        self.python_function = client._build_class(
            PythonFunction, pythonFunction)

    def __repr__(self):
        return f"PipelineStep(pipeline_step_id={repr(self.pipeline_step_id)},\n  pipeline_id={repr(self.pipeline_id)},\n  step_name={repr(self.step_name)},\n  pipeline_name={repr(self.pipeline_name)},\n  created_at={repr(self.created_at)},\n  updated_at={repr(self.updated_at)},\n  python_function_id={repr(self.python_function_id)},\n  step_dependencies={repr(self.step_dependencies)},\n  python_function={repr(self.python_function)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'pipeline_step_id': self.pipeline_step_id, 'pipeline_id': self.pipeline_id, 'step_name': self.step_name, 'pipeline_name': self.pipeline_name, 'created_at': self.created_at, 'updated_at': self.updated_at, 'python_function_id': self.python_function_id, 'step_dependencies': self.step_dependencies, 'python_function': self._get_attribute_as_dict(self.python_function)}
