from .pipeline_reference import PipelineReference
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
            notebookId (str): The reference to the notebook this pipeline belongs to.
            steps (PipelineStep): A list of the pipeline steps attached to the pipeline.
            pipelineReferences (PipelineReference): A list of references from the pipeline to other objects
    """

    def __init__(self, client, pipelineName=None, pipelineId=None, createdAt=None, pipelineVariableMappings=None, notebookId=None, steps={}, pipelineReferences={}):
        super().__init__(client, pipelineId)
        self.pipeline_name = pipelineName
        self.pipeline_id = pipelineId
        self.created_at = createdAt
        self.pipeline_variable_mappings = pipelineVariableMappings
        self.notebook_id = notebookId
        self.steps = client._build_class(PipelineStep, steps)
        self.pipeline_references = client._build_class(
            PipelineReference, pipelineReferences)

    def __repr__(self):
        return f"Pipeline(pipeline_name={repr(self.pipeline_name)},\n  pipeline_id={repr(self.pipeline_id)},\n  created_at={repr(self.created_at)},\n  pipeline_variable_mappings={repr(self.pipeline_variable_mappings)},\n  notebook_id={repr(self.notebook_id)},\n  steps={repr(self.steps)},\n  pipeline_references={repr(self.pipeline_references)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'pipeline_name': self.pipeline_name, 'pipeline_id': self.pipeline_id, 'created_at': self.created_at, 'pipeline_variable_mappings': self.pipeline_variable_mappings, 'notebook_id': self.notebook_id, 'steps': self._get_attribute_as_dict(self.steps), 'pipeline_references': self._get_attribute_as_dict(self.pipeline_references)}
