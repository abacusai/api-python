from .return_class import AbstractApiClass
from .model_monitor_version import ModelMonitorVersion
from .refresh_schedule import RefreshSchedule


class ModelMonitor(AbstractApiClass):
    """
        A model monitor
    """

    def __init__(self, client, name=None, modelMonitorId=None, createdAt=None, projectId=None, modelInstance={}, refreshSchedules={}):
        super().__init__(client, modelMonitorId)
        self.name = name
        self.model_monitor_id = modelMonitorId
        self.created_at = createdAt
        self.project_id = projectId
        self.model_instance = client._build_class(
            ModelMonitorVersion, modelInstance)
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)

    def __repr__(self):
        return f"ModelMonitor(name={repr(self.name)},\n  model_monitor_id={repr(self.model_monitor_id)},\n  created_at={repr(self.created_at)},\n  project_id={repr(self.project_id)},\n  model_instance={repr(self.model_instance)},\n  refresh_schedules={repr(self.refresh_schedules)})"

    def to_dict(self):
        return {'name': self.name, 'model_monitor_id': self.model_monitor_id, 'created_at': self.created_at, 'project_id': self.project_id, 'model_instance': self._get_attribute_as_dict(self.model_instance), 'refresh_schedules': self._get_attribute_as_dict(self.refresh_schedules)}

    def rerun(self):
        return self.client.rerun_model_monitor(self.model_monitor_id)

    def refresh(self):
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        return self.client.describe_model_monitor(self.model_monitor_id)

    def list_versions(self, limit=100, start_after_version=None):
        return self.client.list_model_monitor_versions(self.model_monitor_id, limit, start_after_version)

    def rename(self, name):
        return self.client.rename_model_monitor(self.model_monitor_id, name)

    def delete(self):
        return self.client.delete_model_monitor(self.model_monitor_id)
