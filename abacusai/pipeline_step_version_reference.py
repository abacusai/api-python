from .return_class import AbstractApiClass


class PipelineStepVersionReference(AbstractApiClass):
    """
        A reference from a pipeline step version to the versions that were output from the pipeline step.

        Args:
            client (ApiClient): An authenticated API Client instance
            pipelineStepVersionReferenceId (str): The id of the reference.
            pipelineStepVersion (str): The pipeline step version the reference is connected to.
            objectType (str): The object type of the reference.
            datasetVersion (str): The dataset version the reference is connected to.
            modelVersion (str): The model version the reference is connected to.
            deploymentVersion (str): The deployment version the reference is connected to.
            batchPredictionId (str): The batch prediction id the reference is connected to.
            modelMonitorVersion (str): The model monitor version the reference is connected to.
            notebookVersion (str): The notebook version the reference is connected to.
            featureGroupVersion (str): The feature group version the reference is connected to.
            status (str): The status of the reference
            error (str): The error message if the reference is in an error state.
    """

    def __init__(self, client, pipelineStepVersionReferenceId=None, pipelineStepVersion=None, objectType=None, datasetVersion=None, modelVersion=None, deploymentVersion=None, batchPredictionId=None, modelMonitorVersion=None, notebookVersion=None, featureGroupVersion=None, status=None, error=None):
        super().__init__(client, pipelineStepVersionReferenceId)
        self.pipeline_step_version_reference_id = pipelineStepVersionReferenceId
        self.pipeline_step_version = pipelineStepVersion
        self.object_type = objectType
        self.dataset_version = datasetVersion
        self.model_version = modelVersion
        self.deployment_version = deploymentVersion
        self.batch_prediction_id = batchPredictionId
        self.model_monitor_version = modelMonitorVersion
        self.notebook_version = notebookVersion
        self.feature_group_version = featureGroupVersion
        self.status = status
        self.error = error
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'pipeline_step_version_reference_id': repr(self.pipeline_step_version_reference_id), f'pipeline_step_version': repr(self.pipeline_step_version), f'object_type': repr(self.object_type), f'dataset_version': repr(self.dataset_version), f'model_version': repr(self.model_version), f'deployment_version': repr(
            self.deployment_version), f'batch_prediction_id': repr(self.batch_prediction_id), f'model_monitor_version': repr(self.model_monitor_version), f'notebook_version': repr(self.notebook_version), f'feature_group_version': repr(self.feature_group_version), f'status': repr(self.status), f'error': repr(self.error)}
        class_name = "PipelineStepVersionReference"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'pipeline_step_version_reference_id': self.pipeline_step_version_reference_id, 'pipeline_step_version': self.pipeline_step_version, 'object_type': self.object_type, 'dataset_version': self.dataset_version, 'model_version': self.model_version,
                'deployment_version': self.deployment_version, 'batch_prediction_id': self.batch_prediction_id, 'model_monitor_version': self.model_monitor_version, 'notebook_version': self.notebook_version, 'feature_group_version': self.feature_group_version, 'status': self.status, 'error': self.error}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
