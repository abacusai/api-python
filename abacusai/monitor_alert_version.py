from .return_class import AbstractApiClass


class MonitorAlertVersion(AbstractApiClass):
    """
        A monitor alert version

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The user-friendly name for the monitor alert.
            monitorAlertVersion (str): The identifier for the alert version.
            monitorAlertId (str): The identifier for the alert.
            status (str): The current status of the monitor alert.
            createdAt (str): Date and time at which the monitor alert was created.
            alertingStartedAt (str): The start time and date of the monitor alerting process.
            alertingCompletedAt (str): The end time and date of the monitor alerting process.
            error (str): Relevant error if the status is FAILED.
            modelMonitorVersion (str): The model monitor version associated with the monitor alert version.
            conditionConfig (dict): The condition configuration for this alert.
            actionConfig (dict): The action configuration for this alert.
            alertResult (str): The current result of the alert
            actionStatus (str): The current status of the action as a result of the monitor alert.
            actionError (str): Relevant error if the action status is FAILED.
            actionStartedAt (str): The start time and date of the actionfor the alerting process.
            actionCompletedAt (str): The end time and date of the actionfor the alerting process.
            conditionDescription (str): User friendly description of the condition
            actionDescription (str): User friendly description of the action
    """

    def __init__(self, client, name=None, monitorAlertVersion=None, monitorAlertId=None, status=None, createdAt=None, alertingStartedAt=None, alertingCompletedAt=None, error=None, modelMonitorVersion=None, conditionConfig=None, actionConfig=None, alertResult=None, actionStatus=None, actionError=None, actionStartedAt=None, actionCompletedAt=None, conditionDescription=None, actionDescription=None):
        super().__init__(client, monitorAlertVersion)
        self.name = name
        self.monitor_alert_version = monitorAlertVersion
        self.monitor_alert_id = monitorAlertId
        self.status = status
        self.created_at = createdAt
        self.alerting_started_at = alertingStartedAt
        self.alerting_completed_at = alertingCompletedAt
        self.error = error
        self.model_monitor_version = modelMonitorVersion
        self.condition_config = conditionConfig
        self.action_config = actionConfig
        self.alert_result = alertResult
        self.action_status = actionStatus
        self.action_error = actionError
        self.action_started_at = actionStartedAt
        self.action_completed_at = actionCompletedAt
        self.condition_description = conditionDescription
        self.action_description = actionDescription

    def __repr__(self):
        return f"MonitorAlertVersion(name={repr(self.name)},\n  monitor_alert_version={repr(self.monitor_alert_version)},\n  monitor_alert_id={repr(self.monitor_alert_id)},\n  status={repr(self.status)},\n  created_at={repr(self.created_at)},\n  alerting_started_at={repr(self.alerting_started_at)},\n  alerting_completed_at={repr(self.alerting_completed_at)},\n  error={repr(self.error)},\n  model_monitor_version={repr(self.model_monitor_version)},\n  condition_config={repr(self.condition_config)},\n  action_config={repr(self.action_config)},\n  alert_result={repr(self.alert_result)},\n  action_status={repr(self.action_status)},\n  action_error={repr(self.action_error)},\n  action_started_at={repr(self.action_started_at)},\n  action_completed_at={repr(self.action_completed_at)},\n  condition_description={repr(self.condition_description)},\n  action_description={repr(self.action_description)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'monitor_alert_version': self.monitor_alert_version, 'monitor_alert_id': self.monitor_alert_id, 'status': self.status, 'created_at': self.created_at, 'alerting_started_at': self.alerting_started_at, 'alerting_completed_at': self.alerting_completed_at, 'error': self.error, 'model_monitor_version': self.model_monitor_version, 'condition_config': self.condition_config, 'action_config': self.action_config, 'alert_result': self.alert_result, 'action_status': self.action_status, 'action_error': self.action_error, 'action_started_at': self.action_started_at, 'action_completed_at': self.action_completed_at, 'condition_description': self.condition_description, 'action_description': self.action_description}

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            MonitorAlertVersion: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Describes a given monitor alert version id

        Args:
            monitor_alert_version (str): Unique string identifier for the monitor alert.

        Returns:
            MonitorAlertVersion: An object describing the monitor alert version.
        """
        return self.client.describe_monitor_alert_version(self.monitor_alert_version)

    def wait_for_monitor_alert(self, timeout=1200):
        """
        A waiting call until model monitor version is ready.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'RUNNING'}, timeout=timeout)

    def get_status(self):
        """
        Gets the status of the monitor alert version.

        Returns:
            str: A string describing the status of a monitor alert version (pending, running, complete, etc.).
        """
        return self.describe().status
