from .refresh_policy import RefreshPolicy
from .return_class import AbstractApiClass


class RefreshPipelineRun(AbstractApiClass):
    """
        This keeps track of the overall status of a refresh. A refresh can span multiple resources such as the creation of new dataset versions and the training of a new model version based on them.

        Args:
            client (ApiClient): An authenticated API Client instance
            refreshPipelineRunId (str): The unique identifier for the refresh pipeline run.
            refreshPolicyId (str): Populated when the run was triggered by a refresh policy.
            createdAt (str): The time when this refresh pipeline run was created, in ISO-8601 format.
            startedAt (str): The time when the refresh pipeline run was started, in ISO-8601 format.
            completedAt (str): The time when the refresh pipeline run was completed, in ISO-8601 format.
            status (str): The status of the refresh pipeline run.
            refreshType (str): The type of refresh policy to be run.
            datasetVersions (list[str]): A list of dataset version IDs that this refresh pipeline run is monitoring.
            featureGroupVersion (str): The feature group version ID that this refresh pipeline run is monitoring.
            modelVersions (list[str]): A list of model version IDs that this refresh pipeline run is monitoring.
            predictionMetricVersions (list[str]): A list of prediction metric versions that this refresh pipeline run is monitoring.
            deploymentVersions (list[str]): A list of deployment version IDs that this refresh pipeline run is monitoring.
            batchPredictions (list[str]): A list of batch prediction IDs that this refresh pipeline run is monitoring.
            refreshPolicy (RefreshPolicy): The refresh policy for this refresh policy run.
    """

    def __init__(self, client, refreshPipelineRunId=None, refreshPolicyId=None, createdAt=None, startedAt=None, completedAt=None, status=None, refreshType=None, datasetVersions=None, featureGroupVersion=None, modelVersions=None, predictionMetricVersions=None, deploymentVersions=None, batchPredictions=None, refreshPolicy={}):
        super().__init__(client, refreshPipelineRunId)
        self.refresh_pipeline_run_id = refreshPipelineRunId
        self.refresh_policy_id = refreshPolicyId
        self.created_at = createdAt
        self.started_at = startedAt
        self.completed_at = completedAt
        self.status = status
        self.refresh_type = refreshType
        self.dataset_versions = datasetVersions
        self.feature_group_version = featureGroupVersion
        self.model_versions = modelVersions
        self.prediction_metric_versions = predictionMetricVersions
        self.deployment_versions = deploymentVersions
        self.batch_predictions = batchPredictions
        self.refresh_policy = client._build_class(RefreshPolicy, refreshPolicy)

    def __repr__(self):
        return f"RefreshPipelineRun(refresh_pipeline_run_id={repr(self.refresh_pipeline_run_id)},\n  refresh_policy_id={repr(self.refresh_policy_id)},\n  created_at={repr(self.created_at)},\n  started_at={repr(self.started_at)},\n  completed_at={repr(self.completed_at)},\n  status={repr(self.status)},\n  refresh_type={repr(self.refresh_type)},\n  dataset_versions={repr(self.dataset_versions)},\n  feature_group_version={repr(self.feature_group_version)},\n  model_versions={repr(self.model_versions)},\n  prediction_metric_versions={repr(self.prediction_metric_versions)},\n  deployment_versions={repr(self.deployment_versions)},\n  batch_predictions={repr(self.batch_predictions)},\n  refresh_policy={repr(self.refresh_policy)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'refresh_pipeline_run_id': self.refresh_pipeline_run_id, 'refresh_policy_id': self.refresh_policy_id, 'created_at': self.created_at, 'started_at': self.started_at, 'completed_at': self.completed_at, 'status': self.status, 'refresh_type': self.refresh_type, 'dataset_versions': self.dataset_versions, 'feature_group_version': self.feature_group_version, 'model_versions': self.model_versions, 'prediction_metric_versions': self.prediction_metric_versions, 'deployment_versions': self.deployment_versions, 'batch_predictions': self.batch_predictions, 'refresh_policy': self._get_attribute_as_dict(self.refresh_policy)}

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            RefreshPipelineRun: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Retrieve a single refresh pipeline run

        Args:
            refresh_pipeline_run_id (str): Unique string identifier associated with the refresh pipeline run.

        Returns:
            RefreshPipelineRun: A refresh pipeline run object.
        """
        return self.client.describe_refresh_pipeline_run(self.refresh_pipeline_run_id)

    def wait_for_complete(self, timeout=None):
        """
        A waiting call until refresh pipeline run has completed.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'RUNNING'}, delay=30, timeout=timeout)

    def get_status(self):
        """
        Gets the status of the refresh pipeline run.

        Returns:
            str: A string describing the status of a refresh pipeline run (pending, complete, etc.).
        """
        return self.describe().status
