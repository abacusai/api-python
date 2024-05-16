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
            predictionFeatureGroupVersion (list[str]): Feature group versions that this model monitor is monitoring.
            trainingFeatureGroupVersion (list[str]): Feature group versions that this model monitor is monitoring.
            alertConfig (dict): Alerting configuration for this model monitor.
            biasMetricId (str): The bias metric ID
            metricConfigs (dict): Configurations for model monitor
            featureGroupMonitorConfigs (dict): Configurations for feature group monitor
            metricTypes (dict): List of metric types
            modelId (str): Model ID that this model monitor is monitoring.
            starred (bool): Whether this model monitor is starred.
            batchPredictionId (str): The batch prediction ID this model monitor monitors
            monitorType (str): The type of the monitor, one of MODEL_MONITOR, or FEATURE_GROUP_MONITOR
            edaConfigs (dict): The configs for EDA
            trainingForecastConfig (dict): The tarining config for forecast monitors
            predictionForecastConfig (dict): The prediction config for forecast monitors
            forecastFrequency (str): The frequency of the forecast
            trainingFeatureGroupSampling (bool): Whether or not we sample from training feature group
            predictionFeatureGroupSampling (bool): Whether or not we sample from prediction feature group
            monitorDriftConfig (dict): The monitor drift config for the monitor
            predictionDataUseMappings (dict): The data_use mapping of the prediction features
            trainingDataUseMappings (dict): The data_use mapping of the training features
            latestMonitorModelVersion (ModelMonitorVersion): The latest model monitor version.
            refreshSchedules (RefreshSchedule): List of refresh schedules that indicate when the next model version will be trained.
    """

    def __init__(self, client, modelMonitorId=None, name=None, createdAt=None, projectId=None, trainingFeatureGroupId=None, predictionFeatureGroupId=None, predictionFeatureGroupVersion=None, trainingFeatureGroupVersion=None, alertConfig=None, biasMetricId=None, metricConfigs=None, featureGroupMonitorConfigs=None, metricTypes=None, modelId=None, starred=None, batchPredictionId=None, monitorType=None, edaConfigs=None, trainingForecastConfig=None, predictionForecastConfig=None, forecastFrequency=None, trainingFeatureGroupSampling=None, predictionFeatureGroupSampling=None, monitorDriftConfig=None, predictionDataUseMappings=None, trainingDataUseMappings=None, refreshSchedules={}, latestMonitorModelVersion={}):
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
        self.metric_configs = metricConfigs
        self.feature_group_monitor_configs = featureGroupMonitorConfigs
        self.metric_types = metricTypes
        self.model_id = modelId
        self.starred = starred
        self.batch_prediction_id = batchPredictionId
        self.monitor_type = monitorType
        self.eda_configs = edaConfigs
        self.training_forecast_config = trainingForecastConfig
        self.prediction_forecast_config = predictionForecastConfig
        self.forecast_frequency = forecastFrequency
        self.training_feature_group_sampling = trainingFeatureGroupSampling
        self.prediction_feature_group_sampling = predictionFeatureGroupSampling
        self.monitor_drift_config = monitorDriftConfig
        self.prediction_data_use_mappings = predictionDataUseMappings
        self.training_data_use_mappings = trainingDataUseMappings
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)
        self.latest_monitor_model_version = client._build_class(
            ModelMonitorVersion, latestMonitorModelVersion)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'model_monitor_id': repr(self.model_monitor_id), f'name': repr(self.name), f'created_at': repr(self.created_at), f'project_id': repr(self.project_id), f'training_feature_group_id': repr(self.training_feature_group_id), f'prediction_feature_group_id': repr(self.prediction_feature_group_id), f'prediction_feature_group_version': repr(self.prediction_feature_group_version), f'training_feature_group_version': repr(self.training_feature_group_version), f'alert_config': repr(self.alert_config), f'bias_metric_id': repr(self.bias_metric_id), f'metric_configs': repr(self.metric_configs), f'feature_group_monitor_configs': repr(self.feature_group_monitor_configs), f'metric_types': repr(self.metric_types), f'model_id': repr(self.model_id), f'starred': repr(self.starred), f'batch_prediction_id': repr(
            self.batch_prediction_id), f'monitor_type': repr(self.monitor_type), f'eda_configs': repr(self.eda_configs), f'training_forecast_config': repr(self.training_forecast_config), f'prediction_forecast_config': repr(self.prediction_forecast_config), f'forecast_frequency': repr(self.forecast_frequency), f'training_feature_group_sampling': repr(self.training_feature_group_sampling), f'prediction_feature_group_sampling': repr(self.prediction_feature_group_sampling), f'monitor_drift_config': repr(self.monitor_drift_config), f'prediction_data_use_mappings': repr(self.prediction_data_use_mappings), f'training_data_use_mappings': repr(self.training_data_use_mappings), f'refresh_schedules': repr(self.refresh_schedules), f'latest_monitor_model_version': repr(self.latest_monitor_model_version)}
        class_name = "ModelMonitor"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'model_monitor_id': self.model_monitor_id, 'name': self.name, 'created_at': self.created_at, 'project_id': self.project_id, 'training_feature_group_id': self.training_feature_group_id, 'prediction_feature_group_id': self.prediction_feature_group_id, 'prediction_feature_group_version': self.prediction_feature_group_version, 'training_feature_group_version': self.training_feature_group_version, 'alert_config': self.alert_config, 'bias_metric_id': self.bias_metric_id, 'metric_configs': self.metric_configs, 'feature_group_monitor_configs': self.feature_group_monitor_configs, 'metric_types': self.metric_types, 'model_id': self.model_id, 'starred': self.starred, 'batch_prediction_id': self.batch_prediction_id,
                'monitor_type': self.monitor_type, 'eda_configs': self.eda_configs, 'training_forecast_config': self.training_forecast_config, 'prediction_forecast_config': self.prediction_forecast_config, 'forecast_frequency': self.forecast_frequency, 'training_feature_group_sampling': self.training_feature_group_sampling, 'prediction_feature_group_sampling': self.prediction_feature_group_sampling, 'monitor_drift_config': self.monitor_drift_config, 'prediction_data_use_mappings': self.prediction_data_use_mappings, 'training_data_use_mappings': self.training_data_use_mappings, 'refresh_schedules': self._get_attribute_as_dict(self.refresh_schedules), 'latest_monitor_model_version': self._get_attribute_as_dict(self.latest_monitor_model_version)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

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

    def list_monitor_alerts_for_monitor(self, realtime_monitor_id: str = None):
        """
        Retrieves the list of monitor alerts for a specified monitor. One of the model_monitor_id or realtime_monitor_id is required but not both.

        Args:
            realtime_monitor_id (str): The unique ID associated with the real-time monitor.

        Returns:
            list[MonitorAlert]: A list of monitor alerts.
        """
        return self.client.list_monitor_alerts_for_monitor(self.model_monitor_id, realtime_monitor_id)
