from .model_monitor_version import ModelMonitorVersion
from .refresh_schedule import RefreshSchedule
from .return_class import AbstractApiClass


class ModelMonitor(AbstractApiClass):
    """
        A model monitor

        Args:
            client (ApiClient): An authenticated API Client instance
            modelMonitorId (str): The unique identifier of the model monitor.
            name (str): The user-friendly name for the model monitor.
            createdAt (str): Date and time at which the model was created.
            projectId (str): The project this model belongs to.
            trainingFeatureGroupId (list[str]): Feature group IDs that this model monitor is monitoring.
            predictionFeatureGroupId (list[str]): Feature group IDs that this model monitor is monitoring.
            predictionFeatureGroupVersion (list[str]): Feature group version that this model monitor is monitoring.
            trainingFeatureGroupVersion (list[str]): Feature group version that this model monitor is monitoring.
            alertConfig (dict): Alerting configuration for this model monitor.
            biasMetricId (str): The bias metric ID
            latestBiasMetricVersionId (str): Lastest prediction metric instance for bias
            predictionMetricId (str): The prediction metric ID for the model monitor
            latestPredictionMetricVersionId (str): Lastest prediction metric instance for decile and other analysis
            metricConfigs (dict): Configurations for model monitor
            featureGroupMonitorConfigs (dict): Configurations for feature group monitor
            metricTypes (dict): List of metric types
            modelId (list[str]): Model ID that this model monitor is monitoring.
            starred (bool): Whether this model monitor is starred.
            batchPredictionId (str): The batch prediction ID this model monitor monitors
            monitorType (str): The type of the monitor, one of MODEL_MONITOR, or FEATURE_GROUP_MONITOR
            edaConfigs (dict): The configs for EDA
            latestMonitorModelVersion (ModelMonitorVersion): The latest model monitor version.
            refreshSchedules (RefreshSchedule): List of refresh schedules that indicate when the next model version will be trained.
    """

    def __init__(self, client, modelMonitorId=None, name=None, createdAt=None, projectId=None, trainingFeatureGroupId=None, predictionFeatureGroupId=None, predictionFeatureGroupVersion=None, trainingFeatureGroupVersion=None, alertConfig=None, biasMetricId=None, latestBiasMetricVersionId=None, predictionMetricId=None, latestPredictionMetricVersionId=None, metricConfigs=None, featureGroupMonitorConfigs=None, metricTypes=None, modelId=None, starred=None, batchPredictionId=None, monitorType=None, edaConfigs=None, refreshSchedules={}, latestMonitorModelVersion={}):
        super().__init__(client, modelMonitorId)
        self.model_monitor_id = modelMonitorId
        self.name = name
        self.created_at = createdAt
        self.project_id = projectId
        self.training_feature_group_id = trainingFeatureGroupId
        self.prediction_feature_group_id = predictionFeatureGroupId
        self.prediction_feature_group_version = predictionFeatureGroupVersion
        self.training_feature_group_version = trainingFeatureGroupVersion
        self.alert_config = alertConfig
        self.bias_metric_id = biasMetricId
        self.latest_bias_metric_version_id = latestBiasMetricVersionId
        self.prediction_metric_id = predictionMetricId
        self.latest_prediction_metric_version_id = latestPredictionMetricVersionId
        self.metric_configs = metricConfigs
        self.feature_group_monitor_configs = featureGroupMonitorConfigs
        self.metric_types = metricTypes
        self.model_id = modelId
        self.starred = starred
        self.batch_prediction_id = batchPredictionId
        self.monitor_type = monitorType
        self.eda_configs = edaConfigs
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)
        self.latest_monitor_model_version = client._build_class(
            ModelMonitorVersion, latestMonitorModelVersion)

    def __repr__(self):
        return f"ModelMonitor(model_monitor_id={repr(self.model_monitor_id)},\n  name={repr(self.name)},\n  created_at={repr(self.created_at)},\n  project_id={repr(self.project_id)},\n  training_feature_group_id={repr(self.training_feature_group_id)},\n  prediction_feature_group_id={repr(self.prediction_feature_group_id)},\n  prediction_feature_group_version={repr(self.prediction_feature_group_version)},\n  training_feature_group_version={repr(self.training_feature_group_version)},\n  alert_config={repr(self.alert_config)},\n  bias_metric_id={repr(self.bias_metric_id)},\n  latest_bias_metric_version_id={repr(self.latest_bias_metric_version_id)},\n  prediction_metric_id={repr(self.prediction_metric_id)},\n  latest_prediction_metric_version_id={repr(self.latest_prediction_metric_version_id)},\n  metric_configs={repr(self.metric_configs)},\n  feature_group_monitor_configs={repr(self.feature_group_monitor_configs)},\n  metric_types={repr(self.metric_types)},\n  model_id={repr(self.model_id)},\n  starred={repr(self.starred)},\n  batch_prediction_id={repr(self.batch_prediction_id)},\n  monitor_type={repr(self.monitor_type)},\n  eda_configs={repr(self.eda_configs)},\n  refresh_schedules={repr(self.refresh_schedules)},\n  latest_monitor_model_version={repr(self.latest_monitor_model_version)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'model_monitor_id': self.model_monitor_id, 'name': self.name, 'created_at': self.created_at, 'project_id': self.project_id, 'training_feature_group_id': self.training_feature_group_id, 'prediction_feature_group_id': self.prediction_feature_group_id, 'prediction_feature_group_version': self.prediction_feature_group_version, 'training_feature_group_version': self.training_feature_group_version, 'alert_config': self.alert_config, 'bias_metric_id': self.bias_metric_id, 'latest_bias_metric_version_id': self.latest_bias_metric_version_id, 'prediction_metric_id': self.prediction_metric_id, 'latest_prediction_metric_version_id': self.latest_prediction_metric_version_id, 'metric_configs': self.metric_configs, 'feature_group_monitor_configs': self.feature_group_monitor_configs, 'metric_types': self.metric_types, 'model_id': self.model_id, 'starred': self.starred, 'batch_prediction_id': self.batch_prediction_id, 'monitor_type': self.monitor_type, 'eda_configs': self.eda_configs, 'refresh_schedules': self._get_attribute_as_dict(self.refresh_schedules), 'latest_monitor_model_version': self._get_attribute_as_dict(self.latest_monitor_model_version)}

    def rerun(self):
        """
        Re-runs the specified model monitor.

        Args:
            model_monitor_id (str): Unique string identifier of the model monitor to re-run.

        Returns:
            ModelMonitor: The model monitor that is being re-run.
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
            model_monitor_id (str): Unique string identifier associated with the model monitor.

        Returns:
            ModelMonitor: Description of the model monitor.
        """
        return self.client.describe_model_monitor(self.model_monitor_id)

    def get_summary(self):
        """
        Gets the summary of a model monitor across versions.

        Args:
            model_monitor_id (str): A unique string identifier associated with the model monitor.

        Returns:
            ModelMonitorSummary: An object describing integrity, bias violations, model accuracy and drift for the model monitor.
        """
        return self.client.get_model_monitor_summary(self.model_monitor_id)

    def list_versions(self, limit: int = 100, start_after_version: str = None):
        """
        Retrieves a list of versions for a given model monitor.

        Args:
            limit (int): The maximum length of the list of all model monitor versions.
            start_after_version (str): The ID of the version after which the list starts.

        Returns:
            list[ModelMonitorVersion]: A list of model monitor versions.
        """
        return self.client.list_model_monitor_versions(self.model_monitor_id, limit, start_after_version)

    def rename(self, name: str):
        """
        Renames a model monitor

        Args:
            name (str): The new name to apply to the model monitor.
        """
        return self.client.rename_model_monitor(self.model_monitor_id, name)

    def delete(self):
        """
        Deletes the specified Model Monitor and all its versions.

        Args:
            model_monitor_id (str): Unique identifier of the Model Monitor to delete.
        """
        return self.client.delete_model_monitor(self.model_monitor_id)

    def list_monitor_alerts_for_monitor(self):
        """
        Retrieves the list of monitor alerts for a specified monitor.

        Args:
            model_monitor_id (str): The unique ID associated with the model monitor.

        Returns:
            list[MonitorAlert]: A list of monitor alerts.
        """
        return self.client.list_monitor_alerts_for_monitor(self.model_monitor_id)
