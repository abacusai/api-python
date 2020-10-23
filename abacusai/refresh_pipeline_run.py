from .refresh_policy import RefreshPolicy


class RefreshPipelineRun():
    '''

    '''

    def __init__(self, client, refreshPipelineRunId=None, refreshPolicyId=None, createdAt=None, startedAt=None, completedAt=None, status=None, refreshType=None, datasetVersions=None, modelVersions=None, deploymentVersions=None, batchPredictions=None, refreshPolicy={}):
        self.client = client
        self.id = refreshPipelineRunId
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
        return f"RefreshPipelineRun(refresh_pipeline_run_id={repr(self.refresh_pipeline_run_id)}, refresh_policy_id={repr(self.refresh_policy_id)}, created_at={repr(self.created_at)}, started_at={repr(self.started_at)}, completed_at={repr(self.completed_at)}, status={repr(self.status)}, refresh_type={repr(self.refresh_type)}, dataset_versions={repr(self.dataset_versions)}, model_versions={repr(self.model_versions)}, deployment_versions={repr(self.deployment_versions)}, batch_predictions={repr(self.batch_predictions)}, refresh_policy={repr(self.refresh_policy)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'refresh_pipeline_run_id': self.refresh_pipeline_run_id, 'refresh_policy_id': self.refresh_policy_id, 'created_at': self.created_at, 'started_at': self.started_at, 'completed_at': self.completed_at, 'status': self.status, 'refresh_type': self.refresh_type, 'dataset_versions': self.dataset_versions, 'model_versions': self.model_versions, 'deployment_versions': self.deployment_versions, 'batch_predictions': self.batch_predictions, 'refresh_policy': self.refresh_policy.to_dict() if self.refresh_policy else None}
