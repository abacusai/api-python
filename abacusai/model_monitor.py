from .model_monitor_version import ModelMonitorVersion
from .refresh_schedule import RefreshSchedule
from .return_class import AbstractApiClass


class ModelMonitor(AbstractApiClass):
    """
        A model monitor
    """

    def __init__(self, client, modelMonitorId=None, name=None, createdAt=None, projectId=None, trainingFeatureGroupId=None, predictionFeatureGroupId=None, alertConfig=None, refreshSchedules={}, latestMonitorModelVersion={}):
        super().__init__(client, modelMonitorId)
        self.model_monitor_id = modelMonitorId
        self.name = name
        self.created_at = createdAt
        self.project_id = projectId
        self.training_feature_group_id = trainingFeatureGroupId
        self.prediction_feature_group_id = predictionFeatureGroupId
        self.alert_config = alertConfig
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)
        self.latest_monitor_model_version = client._build_class(
            ModelMonitorVersion, latestMonitorModelVersion)

    def __repr__(self):
        return f"ModelMonitor(model_monitor_id={repr(self.model_monitor_id)},\n  name={repr(self.name)},\n  created_at={repr(self.created_at)},\n  project_id={repr(self.project_id)},\n  training_feature_group_id={repr(self.training_feature_group_id)},\n  prediction_feature_group_id={repr(self.prediction_feature_group_id)},\n  alert_config={repr(self.alert_config)},\n  refresh_schedules={repr(self.refresh_schedules)},\n  latest_monitor_model_version={repr(self.latest_monitor_model_version)})"

    def to_dict(self):
        return {'model_monitor_id': self.model_monitor_id, 'name': self.name, 'created_at': self.created_at, 'project_id': self.project_id, 'training_feature_group_id': self.training_feature_group_id, 'prediction_feature_group_id': self.prediction_feature_group_id, 'alert_config': self.alert_config, 'refresh_schedules': self._get_attribute_as_dict(self.refresh_schedules), 'latest_monitor_model_version': self._get_attribute_as_dict(self.latest_monitor_model_version)}

    def rerun(self):
        """Reruns the specified model monitor."""
        return self.client.rerun_model_monitor(self.model_monitor_id)

    def refresh(self):
        """Calls describe and refreshes the current object's fields"""
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """Retrieves a full description of the specified model monitor."""
        return self.client.describe_model_monitor(self.model_monitor_id)

    def list_versions(self, limit=100, start_after_version=None):
        """Retrieves a list of the versions for a given model monitor."""
        return self.client.list_model_monitor_versions(self.model_monitor_id, limit, start_after_version)

    def rename(self, name):
        """Renames a model monitor"""
        return self.client.rename_model_monitor(self.model_monitor_id, name)

    def delete(self):
        """Deletes the specified model monitor and all its versions."""
        return self.client.delete_model_monitor(self.model_monitor_id)
