from typing import Union

from .api_class import FeatureGroupExportConfig
from .feature_group_refresh_export_config import FeatureGroupRefreshExportConfig
from .return_class import AbstractApiClass


class RefreshPolicy(AbstractApiClass):
    """
        A Refresh Policy describes the frequency at which one or more datasets/models/deployments/batch_predictions can be updated.

        Args:
            client (ApiClient): An authenticated API Client instance
            refreshPolicyId (str): The unique identifier for the refresh policy
            name (str): The user-friendly name for the refresh policy
            cron (str): A cron-style string that describes when this refresh policy is to be executed in UTC
            nextRunTime (str): The next UTC time that this refresh policy will be executed
            createdAt (str): The time when the refresh policy was created
            refreshType (str): The type of refresh policy to be run
            projectId (str): The unique identifier of a project that this refresh policy applies to
            datasetIds (list[str]): Comma-separated list of Dataset IDs that this refresh policy applies to
            featureGroupId (str): Feature Group ID that this refresh policy applies to
            modelIds (list[str]): Comma-separated list of Model IDs that this refresh policy applies to
            deploymentIds (list[str]): Comma-separated list of Deployment IDs that this refresh policy applies to
            predictionMetricIds (list[str]): Comma-separated list of Prediction Metric IDs that this refresh policy applies to
            modelMonitorIds (list[str]): Comma-separated list of Model Monitor IDs that this refresh policy applies to
            notebookId (str): Notebook ID that this refresh policy applies to
            paused (bool): True if the refresh policy is paused
            featureGroupExportConfig (FeatureGroupRefreshExportConfig): The export configuration for the feature group. Only applicable if refresh_type is FEATUREGROUP.
    """

    def __init__(self, client, refreshPolicyId=None, name=None, cron=None, nextRunTime=None, createdAt=None, refreshType=None, projectId=None, datasetIds=None, featureGroupId=None, modelIds=None, deploymentIds=None, predictionMetricIds=None, modelMonitorIds=None, notebookId=None, paused=None, featureGroupExportConfig={}):
        super().__init__(client, refreshPolicyId)
        self.refresh_policy_id = refreshPolicyId
        self.name = name
        self.cron = cron
        self.next_run_time = nextRunTime
        self.created_at = createdAt
        self.refresh_type = refreshType
        self.project_id = projectId
        self.dataset_ids = datasetIds
        self.feature_group_id = featureGroupId
        self.model_ids = modelIds
        self.deployment_ids = deploymentIds
        self.prediction_metric_ids = predictionMetricIds
        self.model_monitor_ids = modelMonitorIds
        self.notebook_id = notebookId
        self.paused = paused
        self.feature_group_export_config = client._build_class(
            FeatureGroupRefreshExportConfig, featureGroupExportConfig)

    def __repr__(self):
        return f"RefreshPolicy(refresh_policy_id={repr(self.refresh_policy_id)},\n  name={repr(self.name)},\n  cron={repr(self.cron)},\n  next_run_time={repr(self.next_run_time)},\n  created_at={repr(self.created_at)},\n  refresh_type={repr(self.refresh_type)},\n  project_id={repr(self.project_id)},\n  dataset_ids={repr(self.dataset_ids)},\n  feature_group_id={repr(self.feature_group_id)},\n  model_ids={repr(self.model_ids)},\n  deployment_ids={repr(self.deployment_ids)},\n  prediction_metric_ids={repr(self.prediction_metric_ids)},\n  model_monitor_ids={repr(self.model_monitor_ids)},\n  notebook_id={repr(self.notebook_id)},\n  paused={repr(self.paused)},\n  feature_group_export_config={repr(self.feature_group_export_config)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'refresh_policy_id': self.refresh_policy_id, 'name': self.name, 'cron': self.cron, 'next_run_time': self.next_run_time, 'created_at': self.created_at, 'refresh_type': self.refresh_type, 'project_id': self.project_id, 'dataset_ids': self.dataset_ids, 'feature_group_id': self.feature_group_id, 'model_ids': self.model_ids, 'deployment_ids': self.deployment_ids, 'prediction_metric_ids': self.prediction_metric_ids, 'model_monitor_ids': self.model_monitor_ids, 'notebook_id': self.notebook_id, 'paused': self.paused, 'feature_group_export_config': self._get_attribute_as_dict(self.feature_group_export_config)}

    def delete(self):
        """
        Delete a refresh policy.

        Args:
            refresh_policy_id (str): Unique string identifier associated with the refresh policy to delete.
        """
        return self.client.delete_refresh_policy(self.refresh_policy_id)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            RefreshPolicy: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Retrieve a single refresh policy

        Args:
            refresh_policy_id (str): The unique ID associated with this refresh policy.

        Returns:
            RefreshPolicy: An object representing the refresh policy.
        """
        return self.client.describe_refresh_policy(self.refresh_policy_id)

    def list_refresh_pipeline_runs(self):
        """
        List the the times that the refresh policy has been run

        Args:
            refresh_policy_id (str): Unique identifier associated with the refresh policy.

        Returns:
            list[RefreshPipelineRun]: List of refresh pipeline runs for the given refresh policy ID.
        """
        return self.client.list_refresh_pipeline_runs(self.refresh_policy_id)

    def pause(self):
        """
        Pauses a refresh policy

        Args:
            refresh_policy_id (str): Unique identifier associated with the refresh policy to be paused.
        """
        return self.client.pause_refresh_policy(self.refresh_policy_id)

    def resume(self):
        """
        Resumes a refresh policy

        Args:
            refresh_policy_id (str): The unique ID associated with this refresh policy.
        """
        return self.client.resume_refresh_policy(self.refresh_policy_id)

    def run(self):
        """
        Force a run of the refresh policy.

        Args:
            refresh_policy_id (str): Unique string identifier associated with the refresh policy to be run.
        """
        return self.client.run_refresh_policy(self.refresh_policy_id)

    def update(self, name: str = None, cron: str = None, feature_group_export_config: Union[dict, FeatureGroupExportConfig] = None):
        """
        Update the name or cron string of a refresh policy

        Args:
            name (str): Name of the refresh policy to be updated.
            cron (str): Cron string describing the schedule from the refresh policy to be updated.
            feature_group_export_config (FeatureGroupExportConfig): Feature group export configuration to update a feature group refresh policy.

        Returns:
            RefreshPolicy: Updated refresh policy.
        """
        return self.client.update_refresh_policy(self.refresh_policy_id, name, cron, feature_group_export_config)
