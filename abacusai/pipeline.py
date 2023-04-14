from .pipeline_step import PipelineStep
from .return_class import AbstractApiClass


class Pipeline(AbstractApiClass):
    """
        A Pipeline For Steps.

        Args:
            client (ApiClient): An authenticated API Client instance
            pipelineName (str): The name of the pipeline this step is a part of.
            pipelineId (str): The reference to the pipeline this step belongs to.
            createdAt (str): The date and time which the pipeline was created.
            pipelineVariableMappings (dict): A description of the function variables into the pipeline.
            steps (PipelineStep): A list of the pipeline steps attached to the pipeline.
    """

    def __init__(self, client, pipelineName=None, pipelineId=None, createdAt=None, pipelineVariableMappings=None, steps={}):
        super().__init__(client, pipelineId)
        self.pipeline_name = pipelineName
        self.pipeline_id = pipelineId
        self.created_at = createdAt
        self.pipeline_variable_mappings = pipelineVariableMappings
        self.steps = client._build_class(PipelineStep, steps)

    def __repr__(self):
        return f"Pipeline(pipeline_name={repr(self.pipeline_name)},\n  pipeline_id={repr(self.pipeline_id)},\n  created_at={repr(self.created_at)},\n  pipeline_variable_mappings={repr(self.pipeline_variable_mappings)},\n  steps={repr(self.steps)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'pipeline_name': self.pipeline_name, 'pipeline_id': self.pipeline_id, 'created_at': self.created_at, 'pipeline_variable_mappings': self.pipeline_variable_mappings, 'steps': self._get_attribute_as_dict(self.steps)}
