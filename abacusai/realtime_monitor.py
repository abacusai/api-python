from .return_class import AbstractApiClass


class RealtimeMonitor(AbstractApiClass):
    """
        A real-time monitor

        Args:
            client (ApiClient): An authenticated API Client instance
            realtimeMonitorId (str): The unique identifier of the real-time monitor.
            name (str): The user-friendly name for the real-time monitor.
            createdAt (str): Date and time at which the real-time monitor was created.
            deploymentId (str): Deployment ID that this real-time monitor is monitoring.
            lookbackTime (int): The lookback time for the real-time monitor.
            realtimeMonitorSchedule (str): The drift computation schedule for the real-time monitor.
    """

    def __init__(self, client, realtimeMonitorId=None, name=None, createdAt=None, deploymentId=None, lookbackTime=None, realtimeMonitorSchedule=None):
        super().__init__(client, realtimeMonitorId)
        self.realtime_monitor_id = realtimeMonitorId
        self.name = name
        self.created_at = createdAt
        self.deployment_id = deploymentId
        self.lookback_time = lookbackTime
        self.realtime_monitor_schedule = realtimeMonitorSchedule
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'realtime_monitor_id': repr(self.realtime_monitor_id), f'name': repr(self.name), f'created_at': repr(self.created_at), f'deployment_id': repr(
            self.deployment_id), f'lookback_time': repr(self.lookback_time), f'realtime_monitor_schedule': repr(self.realtime_monitor_schedule)}
        class_name = "RealtimeMonitor"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'realtime_monitor_id': self.realtime_monitor_id, 'name': self.name, 'created_at': self.created_at,
                'deployment_id': self.deployment_id, 'lookback_time': self.lookback_time, 'realtime_monitor_schedule': self.realtime_monitor_schedule}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def update(self, realtime_monitor_schedule: str = None, lookback_time: float = None):
        """
        Update the real-time monitor associated with the real-time monitor id.

        Args:
            realtime_monitor_schedule (str): The cron expression for triggering monitor
            lookback_time (float): Lookback time (in seconds) for each monitor trigger

        Returns:
            RealtimeMonitor: Object describing the realtime monitor.
        """
        return self.client.update_realtime_monitor(self.realtime_monitor_id, realtime_monitor_schedule, lookback_time)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            RealtimeMonitor: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Get the real-time monitor associated with the real-time monitor id.

        Args:
            realtime_monitor_id (str): Unique string identifier for the real-time monitor.

        Returns:
            RealtimeMonitor: Object describing the real-time monitor.
        """
        return self.client.describe_realtime_monitor(self.realtime_monitor_id)

    def delete(self):
        """
        Delete the real-time monitor associated with the real-time monitor id.

        Args:
            realtime_monitor_id (str): Unique string identifier for the real-time monitor.
        """
        return self.client.delete_realtime_monitor(self.realtime_monitor_id)
