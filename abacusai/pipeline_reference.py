from .return_class import AbstractApiClass


class PipelineReference(AbstractApiClass):
    """
        A reference to a pipeline to the objects it is run on.

        Args:
            client (ApiClient): An authenticated API Client instance
            pipelineReferenceId (str): The id of the reference.
            pipelineId (str): The id of the pipeline for the reference.
            objectType (str): The object type of the reference.
            datasetId (str): The dataset id of the reference.
            modelId (str): The model id of the reference.
            deploymentId (str): The deployment id of the reference.
            batchPredictionDescriptionId (str): The batch prediction description id of the reference.
            modelMonitorId (str): The model monitor id of the reference.
            notebookId (str): The notebook id of the reference.
            featureGroupId (str): The feature group id of the reference.
    """

    def __init__(self, client, pipelineReferenceId=None, pipelineId=None, objectType=None, datasetId=None, modelId=None, deploymentId=None, batchPredictionDescriptionId=None, modelMonitorId=None, notebookId=None, featureGroupId=None):
        super().__init__(client, pipelineReferenceId)
        self.pipeline_reference_id = pipelineReferenceId
        self.pipeline_id = pipelineId
        self.object_type = objectType
        self.dataset_id = datasetId
        self.model_id = modelId
        self.deployment_id = deploymentId
        self.batch_prediction_description_id = batchPredictionDescriptionId
        self.model_monitor_id = modelMonitorId
        self.notebook_id = notebookId
        self.feature_group_id = featureGroupId

    def __repr__(self):
        return f"PipelineReference(pipeline_reference_id={repr(self.pipeline_reference_id)},\n  pipeline_id={repr(self.pipeline_id)},\n  object_type={repr(self.object_type)},\n  dataset_id={repr(self.dataset_id)},\n  model_id={repr(self.model_id)},\n  deployment_id={repr(self.deployment_id)},\n  batch_prediction_description_id={repr(self.batch_prediction_description_id)},\n  model_monitor_id={repr(self.model_monitor_id)},\n  notebook_id={repr(self.notebook_id)},\n  feature_group_id={repr(self.feature_group_id)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'pipeline_reference_id': self.pipeline_reference_id, 'pipeline_id': self.pipeline_id, 'object_type': self.object_type, 'dataset_id': self.dataset_id, 'model_id': self.model_id, 'deployment_id': self.deployment_id, 'batch_prediction_description_id': self.batch_prediction_description_id, 'model_monitor_id': self.model_monitor_id, 'notebook_id': self.notebook_id, 'feature_group_id': self.feature_group_id}
