from .pipeline_reference import PipelineReference
from .pipeline_step import PipelineStep
from .pipeline_version import PipelineVersion
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
            cron (str): A cron-style string that describes when this refresh policy is to be executed in UTC
            nextRunTime (str): The next time this pipeline will be run.
            steps (PipelineStep): A list of the pipeline steps attached to the pipeline.
            pipelineReferences (PipelineReference): A list of references from the pipeline to other objects
            latestPipelineVersion (PipelineVersion): The latest version of the pipeline.
    """

    def __init__(self, client, pipelineName=None, pipelineId=None, createdAt=None, pipelineVariableMappings=None, notebookId=None, cron=None, nextRunTime=None, steps={}, pipelineReferences={}, latestPipelineVersion={}):
        super().__init__(client, pipelineId)
        self.pipeline_name = pipelineName
        self.pipeline_id = pipelineId
        self.created_at = createdAt
        self.pipeline_variable_mappings = pipelineVariableMappings
        self.notebook_id = notebookId
        self.cron = cron
        self.next_run_time = nextRunTime
        self.steps = client._build_class(PipelineStep, steps)
        self.pipeline_references = client._build_class(
            PipelineReference, pipelineReferences)
        self.latest_pipeline_version = client._build_class(
            PipelineVersion, latestPipelineVersion)

    def __repr__(self):
        return f"Pipeline(pipeline_name={repr(self.pipeline_name)},\n  pipeline_id={repr(self.pipeline_id)},\n  created_at={repr(self.created_at)},\n  pipeline_variable_mappings={repr(self.pipeline_variable_mappings)},\n  notebook_id={repr(self.notebook_id)},\n  cron={repr(self.cron)},\n  next_run_time={repr(self.next_run_time)},\n  steps={repr(self.steps)},\n  pipeline_references={repr(self.pipeline_references)},\n  latest_pipeline_version={repr(self.latest_pipeline_version)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'pipeline_name': self.pipeline_name, 'pipeline_id': self.pipeline_id, 'created_at': self.created_at, 'pipeline_variable_mappings': self.pipeline_variable_mappings, 'notebook_id': self.notebook_id, 'cron': self.cron, 'next_run_time': self.next_run_time, 'steps': self._get_attribute_as_dict(self.steps), 'pipeline_references': self._get_attribute_as_dict(self.pipeline_references), 'latest_pipeline_version': self._get_attribute_as_dict(self.latest_pipeline_version)}

    def unset_refresh_schedule(self):
        """
        Deletes the refresh schedule for a given pipeline.

        Args:
            pipeline_id (str): The id of the pipeline.

        Returns:
            Pipeline: Object describing the pipeline.
        """
        return self.client.unset_pipeline_refresh_schedule(self.pipeline_id)

    def pause_refresh_schedule(self):
        """
        Pauses the refresh schedule for a given pipeline.

        Args:
            pipeline_id (str): The id of the pipeline.

        Returns:
            Pipeline: Object describing the pipeline.
        """
        return self.client.pause_pipeline_refresh_schedule(self.pipeline_id)

    def resume_refresh_schedule(self):
        """
        Resumes the refresh schedule for a given pipeline.

        Args:
            pipeline_id (str): The id of the pipeline.

        Returns:
            Pipeline: Object describing the pipeline.
        """
        return self.client.resume_pipeline_refresh_schedule(self.pipeline_id)
