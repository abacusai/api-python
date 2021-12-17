from .refresh_policy import RefreshPolicy
from .return_class import AbstractApiClass


class RefreshPipelineRun(AbstractApiClass):
    """
        A refresh policy run or a project refresh run.
    """

    def __init__(self, client, refreshPipelineRunId=None, refreshPolicyId=None, createdAt=None, startedAt=None, completedAt=None, status=None, refreshType=None, datasetVersions=None, modelVersions=None, deploymentVersions=None, batchPredictions=None, refreshPolicy={}):
        super().__init__(client, refreshPipelineRunId)
        self.refresh_pipeline_run_id = refreshPipelineRunId
        self.refresh_policy_id = refreshPolicyId
        self.created_at = createdAt
        self.started_at = startedAt
        self.completed_at = completedAt
        self.status = status
        self.refresh_type = refreshType
        self.dataset_versions = datasetVersions
        self.model_versions = modelVersions
        self.deployment_versions = deploymentVersions
        self.batch_predictions = batchPredictions
        self.refresh_policy = client._build_class(RefreshPolicy, refreshPolicy)

    def __repr__(self):
        return f"RefreshPipelineRun(refresh_pipeline_run_id={repr(self.refresh_pipeline_run_id)},\n  refresh_policy_id={repr(self.refresh_policy_id)},\n  created_at={repr(self.created_at)},\n  started_at={repr(self.started_at)},\n  completed_at={repr(self.completed_at)},\n  status={repr(self.status)},\n  refresh_type={repr(self.refresh_type)},\n  dataset_versions={repr(self.dataset_versions)},\n  model_versions={repr(self.model_versions)},\n  deployment_versions={repr(self.deployment_versions)},\n  batch_predictions={repr(self.batch_predictions)},\n  refresh_policy={repr(self.refresh_policy)})"

    def to_dict(self):
        return {'refresh_pipeline_run_id': self.refresh_pipeline_run_id, 'refresh_policy_id': self.refresh_policy_id, 'created_at': self.created_at, 'started_at': self.started_at, 'completed_at': self.completed_at, 'status': self.status, 'refresh_type': self.refresh_type, 'dataset_versions': self.dataset_versions, 'model_versions': self.model_versions, 'deployment_versions': self.deployment_versions, 'batch_predictions': self.batch_predictions, 'refresh_policy': self._get_attribute_as_dict(self.refresh_policy)}

    def refresh(self):
        """Calls describe and refreshes the current object's fields"""
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """Retrieve a single refresh pipeline run"""
        return self.client.describe_refresh_pipeline_run(self.refresh_pipeline_run_id)
