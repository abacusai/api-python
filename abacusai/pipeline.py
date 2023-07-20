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
            isProd (bool): Whether this pipeline is a production pipeline.
            steps (PipelineStep): A list of the pipeline steps attached to the pipeline.
            pipelineReferences (PipelineReference): A list of references from the pipeline to other objects
            latestPipelineVersion (PipelineVersion): The latest version of the pipeline.
    """

    def __init__(self, client, pipelineName=None, pipelineId=None, createdAt=None, pipelineVariableMappings=None, notebookId=None, cron=None, nextRunTime=None, isProd=None, steps={}, pipelineReferences={}, latestPipelineVersion={}):
        super().__init__(client, pipelineId)
        self.pipeline_name = pipelineName
        self.pipeline_id = pipelineId
        self.created_at = createdAt
        self.pipeline_variable_mappings = pipelineVariableMappings
        self.notebook_id = notebookId
        self.cron = cron
        self.next_run_time = nextRunTime
        self.is_prod = isProd
        self.steps = client._build_class(PipelineStep, steps)
        self.pipeline_references = client._build_class(
            PipelineReference, pipelineReferences)
        self.latest_pipeline_version = client._build_class(
            PipelineVersion, latestPipelineVersion)

    def __repr__(self):
        return f"Pipeline(pipeline_name={repr(self.pipeline_name)},\n  pipeline_id={repr(self.pipeline_id)},\n  created_at={repr(self.created_at)},\n  pipeline_variable_mappings={repr(self.pipeline_variable_mappings)},\n  notebook_id={repr(self.notebook_id)},\n  cron={repr(self.cron)},\n  next_run_time={repr(self.next_run_time)},\n  is_prod={repr(self.is_prod)},\n  steps={repr(self.steps)},\n  pipeline_references={repr(self.pipeline_references)},\n  latest_pipeline_version={repr(self.latest_pipeline_version)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'pipeline_name': self.pipeline_name, 'pipeline_id': self.pipeline_id, 'created_at': self.created_at, 'pipeline_variable_mappings': self.pipeline_variable_mappings, 'notebook_id': self.notebook_id, 'cron': self.cron, 'next_run_time': self.next_run_time, 'is_prod': self.is_prod, 'steps': self._get_attribute_as_dict(self.steps), 'pipeline_references': self._get_attribute_as_dict(self.pipeline_references), 'latest_pipeline_version': self._get_attribute_as_dict(self.latest_pipeline_version)}

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            Pipeline: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Describes a given pipeline.

        Args:
            pipeline_id (str): The ID of the pipeline to describe.

        Returns:
            Pipeline: An object describing a Pipeline
        """
        return self.client.describe_pipeline(self.pipeline_id)

    def update(self, project_id: str = None, pipeline_variable_mappings: list = None, cron: str = None, is_prod: bool = None):
        """
        Updates a pipeline for executing multiple steps.

        Args:
            project_id (str): A unique string identifier for the pipeline.
            pipeline_variable_mappings (list): List of Python function arguments for the pipeline.
            cron (str): A cron-like string specifying the frequency of the scheduled pipeline runs.
            is_prod (bool): Whether the pipeline is a production pipeline or not.

        Returns:
            Pipeline: An object that describes a Pipeline.
        """
        return self.client.update_pipeline(self.pipeline_id, project_id, pipeline_variable_mappings, cron, is_prod)

    def delete(self):
        """
        Deletes a pipeline.

        Args:
            pipeline_id (str): The ID of the pipeline to delete.
        """
        return self.client.delete_pipeline(self.pipeline_id)

    def list_versions(self, limit: int = 200):
        """
        Lists the pipeline versions for a specified pipeline

        Args:
            limit (int): The maximum number of pipeline versions to return.

        Returns:
            list[PipelineVersion]: A list of pipeline versions.
        """
        return self.client.list_pipeline_versions(self.pipeline_id, limit)

    def run(self, pipeline_variable_mappings: list = None):
        """
        Runs a specified pipeline with the arguments provided.

        Args:
            pipeline_variable_mappings (list): List of Python function arguments for the pipeline.

        Returns:
            PipelineVersion: The object describing the pipeline
        """
        return self.client.run_pipeline(self.pipeline_id, pipeline_variable_mappings)

    def create_step(self, step_name: str, function_name: str = None, source_code: str = None, step_input_mappings: list = None, output_variable_mappings: list = None, step_dependencies: list = None, package_requirements: list = None, cpu_size: str = None, memory: int = None):
        """
        Creates a step in a given pipeline.

        Args:
            step_name (str): The name of the step.
            function_name (str): The name of the Python function.
            source_code (str): Contents of a valid Python source code file. The source code should contain the transform feature group functions. A list of allowed imports and system libraries for each language is specified in the user functions documentation section.
            step_input_mappings (list): List of Python function arguments.
            output_variable_mappings (list): List of Python function ouputs.
            step_dependencies (list): List of step names this step depends on.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            cpu_size (str): Size of the CPU for the step function.
            memory (int): Memory (in GB) for the step function.

        Returns:
            Pipeline: Object describing the pipeline.
        """
        return self.client.create_pipeline_step(self.pipeline_id, step_name, function_name, source_code, step_input_mappings, output_variable_mappings, step_dependencies, package_requirements, cpu_size, memory)

    def describe_step_by_name(self, step_name: str):
        """
        Describes a pipeline step by the step name.

        Args:
            step_name (str): The name of the step.

        Returns:
            PipelineStep: An object describing the pipeline step.
        """
        return self.client.describe_pipeline_step_by_name(self.pipeline_id, step_name)

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
