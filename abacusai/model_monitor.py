from .model_monitor_version import ModelMonitorVersion
from .refresh_schedule import RefreshSchedule
from .return_class import AbstractApiClass


class ModelMonitor(AbstractApiClass):
    """
        A model monitor

        Args:
            client (ApiClient): An authenticated API Client instance
            modelMonitorId (str): The unique identifier of the model monitor.
            name (str): The user-friendly name for the model.
            createdAt (str): Date and time at which the model was created.
            projectId (str): The project this model belongs to.
            trainingFeatureGroupId (unique string identifiers): Feature group IDs that this model monitor is monitoring.
            predictionFeatureGroupId (unique string identifiers): Feature group IDs that this model monitor is monitoring.
            alertConfig (dict): Alerting configuration for this model monitor.
            biasMetricId (str): 
            latestBiasMetricVersionId (str): Lastest prediction metric instance for bias
            predictionMetricId (str): 
            latestPredictionMetricVersionId (str): Lastest prediction metric instance for decile and other analysis
            latestMonitorModelVersion (ModelMonitorVersion): The latest model monitor version.
            refreshSchedules (RefreshSchedule): List of refresh schedules that indicate when the next model version will be trained.
    """

    def __init__(self, client, modelMonitorId=None, name=None, createdAt=None, projectId=None, trainingFeatureGroupId=None, predictionFeatureGroupId=None, alertConfig=None, biasMetricId=None, latestBiasMetricVersionId=None, predictionMetricId=None, latestPredictionMetricVersionId=None, refreshSchedules={}, latestMonitorModelVersion={}):
        super().__init__(client, modelMonitorId)
        self.model_monitor_id = modelMonitorId
        self.name = name
        self.created_at = createdAt
        self.project_id = projectId
        self.training_feature_group_id = trainingFeatureGroupId
        self.prediction_feature_group_id = predictionFeatureGroupId
        self.alert_config = alertConfig
        self.bias_metric_id = biasMetricId
        self.latest_bias_metric_version_id = latestBiasMetricVersionId
        self.prediction_metric_id = predictionMetricId
        self.latest_prediction_metric_version_id = latestPredictionMetricVersionId
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)
        self.latest_monitor_model_version = client._build_class(
            ModelMonitorVersion, latestMonitorModelVersion)

    def __repr__(self):
        return f"ModelMonitor(model_monitor_id={repr(self.model_monitor_id)},\n  name={repr(self.name)},\n  created_at={repr(self.created_at)},\n  project_id={repr(self.project_id)},\n  training_feature_group_id={repr(self.training_feature_group_id)},\n  prediction_feature_group_id={repr(self.prediction_feature_group_id)},\n  alert_config={repr(self.alert_config)},\n  bias_metric_id={repr(self.bias_metric_id)},\n  latest_bias_metric_version_id={repr(self.latest_bias_metric_version_id)},\n  prediction_metric_id={repr(self.prediction_metric_id)},\n  latest_prediction_metric_version_id={repr(self.latest_prediction_metric_version_id)},\n  refresh_schedules={repr(self.refresh_schedules)},\n  latest_monitor_model_version={repr(self.latest_monitor_model_version)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'model_monitor_id': self.model_monitor_id, 'name': self.name, 'created_at': self.created_at, 'project_id': self.project_id, 'training_feature_group_id': self.training_feature_group_id, 'prediction_feature_group_id': self.prediction_feature_group_id, 'alert_config': self.alert_config, 'bias_metric_id': self.bias_metric_id, 'latest_bias_metric_version_id': self.latest_bias_metric_version_id, 'prediction_metric_id': self.prediction_metric_id, 'latest_prediction_metric_version_id': self.latest_prediction_metric_version_id, 'refresh_schedules': self._get_attribute_as_dict(self.refresh_schedules), 'latest_monitor_model_version': self._get_attribute_as_dict(self.latest_monitor_model_version)}

    def rerun(self):
        """
        Reruns the specified model monitor.

        Args:
            model_monitor_id (str): The model monitor to rerun.

        Returns:
            ModelMonitor: The model monitor that is being rerun.
        """
        return self.client.rerun_model_monitor(self.model_monitor_id)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            ModelMonitor: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Retrieves a full description of the specified model monitor.

        Args:
            model_monitor_id (str): The unique ID associated with the model monitor.

        Returns:
            ModelMonitor: The description of the model monitor.
        """
        return self.client.describe_model_monitor(self.model_monitor_id)

    def list_versions(self, limit: int = 100, start_after_version: str = None):
        """
        Retrieves a list of the versions for a given model monitor.

        Args:
            limit (int): The max length of the list of all model monitor versions.
            start_after_version (str): The id of the version after which the list starts.

        Returns:
            ModelMonitorVersion: An array of model monitor versions.
        """
        return self.client.list_model_monitor_versions(self.model_monitor_id, limit, start_after_version)

    def rename(self, name: str):
        """
        Renames a model monitor

        Args:
            name (str): The name to apply to the model monitor
        """
        return self.client.rename_model_monitor(self.model_monitor_id, name)

    def delete(self):
        """
        Deletes the specified model monitor and all its versions.

        Args:
            model_monitor_id (str): The ID of the model monitor to delete.
        """
        return self.client.delete_model_monitor(self.model_monitor_id)
