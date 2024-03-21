from typing import Union

from .api_class import AlertActionConfig, AlertConditionConfig
from .monitor_alert_version import MonitorAlertVersion
from .return_class import AbstractApiClass


class MonitorAlert(AbstractApiClass):
    """
        A Monitor Alert

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The user-friendly name for the alert.
            monitorAlertId (str): The unique identifier of the monitor alert.
            createdAt (str): Date and time at which the monitor alert was created.
            projectId (str): The project this alert belongs to.
            modelMonitorId (str): The monitor id that this alert is associated with
            realtimeMonitorId (str): The realtime monitor id that this alert is associated with
            conditionConfig (dict): The condition configuration for this alert.
            actionConfig (dict): The action configuration for this alert.
            conditionDescription (str): User friendly description of the condition
            actionDescription (str): User friendly description of the action
            alertType (str): The type of the alert
            deploymentId (str): The deployment ID this alert is associated with
            latestMonitorAlertVersion (MonitorAlertVersion): The latest monitor alert version.
    """

    def __init__(self, client, name=None, monitorAlertId=None, createdAt=None, projectId=None, modelMonitorId=None, realtimeMonitorId=None, conditionConfig=None, actionConfig=None, conditionDescription=None, actionDescription=None, alertType=None, deploymentId=None, latestMonitorAlertVersion={}):
        super().__init__(client, monitorAlertId)
        self.name = name
        self.monitor_alert_id = monitorAlertId
        self.created_at = createdAt
        self.project_id = projectId
        self.model_monitor_id = modelMonitorId
        self.realtime_monitor_id = realtimeMonitorId
        self.condition_config = conditionConfig
        self.action_config = actionConfig
        self.condition_description = conditionDescription
        self.action_description = actionDescription
        self.alert_type = alertType
        self.deployment_id = deploymentId
        self.latest_monitor_alert_version = client._build_class(
            MonitorAlertVersion, latestMonitorAlertVersion)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'monitor_alert_id': repr(self.monitor_alert_id), f'created_at': repr(self.created_at), f'project_id': repr(self.project_id), f'model_monitor_id': repr(self.model_monitor_id), f'realtime_monitor_id': repr(self.realtime_monitor_id), f'condition_config': repr(
            self.condition_config), f'action_config': repr(self.action_config), f'condition_description': repr(self.condition_description), f'action_description': repr(self.action_description), f'alert_type': repr(self.alert_type), f'deployment_id': repr(self.deployment_id), f'latest_monitor_alert_version': repr(self.latest_monitor_alert_version)}
        class_name = "MonitorAlert"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'monitor_alert_id': self.monitor_alert_id, 'created_at': self.created_at, 'project_id': self.project_id, 'model_monitor_id': self.model_monitor_id, 'realtime_monitor_id': self.realtime_monitor_id, 'condition_config': self.condition_config,
                'action_config': self.action_config, 'condition_description': self.condition_description, 'action_description': self.action_description, 'alert_type': self.alert_type, 'deployment_id': self.deployment_id, 'latest_monitor_alert_version': self._get_attribute_as_dict(self.latest_monitor_alert_version)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def update(self, alert_name: str = None, condition_config: Union[dict, AlertConditionConfig] = None, action_config: Union[dict, AlertActionConfig] = None):
        """
        Update monitor alert

        Args:
            alert_name (str): Name of the alert.
            condition_config (AlertConditionConfig): Condition to run the actions for the alert.
            action_config (AlertActionConfig): Configuration for the action of the alert.

        Returns:
            MonitorAlert: Object describing the monitor alert.
        """
        return self.client.update_monitor_alert(self.monitor_alert_id, alert_name, condition_config, action_config)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            MonitorAlert: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Describes a given monitor alert id

        Args:
            monitor_alert_id (str): Unique identifier of the monitor alert.

        Returns:
            MonitorAlert: Object containing information about the monitor alert.
        """
        return self.client.describe_monitor_alert(self.monitor_alert_id)

    def run(self):
        """
        Reruns a given monitor alert from latest monitor instance

        Args:
            monitor_alert_id (str): Unique identifier of a monitor alert.

        Returns:
            MonitorAlert: Object describing the monitor alert.
        """
        return self.client.run_monitor_alert(self.monitor_alert_id)

    def delete(self):
        """
        Delets a monitor alert

        Args:
            monitor_alert_id (str): The unique string identifier of the alert to delete.
        """
        return self.client.delete_monitor_alert(self.monitor_alert_id)
