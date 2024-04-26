from .return_class import AbstractApiClass


class RefreshSchedule(AbstractApiClass):
    """
        A refresh schedule for an object. Defines when the next version of the object will be created

        Args:
            client (ApiClient): An authenticated API Client instance
            refreshPolicyId (str): The unique identifier of the refresh policy
            nextRunTime (str): The next run time of the refresh policy. If null, the policy is paused.
            cron (str): A cron-style string that describes the when this refresh policy is to be executed in UTC
            refreshType (str): The type of refresh that will be run
            error (str): An error message for the last pipeline run of a policy
    """

    def __init__(self, client, refreshPolicyId=None, nextRunTime=None, cron=None, refreshType=None, error=None):
        super().__init__(client, None)
        self.refresh_policy_id = refreshPolicyId
        self.next_run_time = nextRunTime
        self.cron = cron
        self.refresh_type = refreshType
        self.error = error
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'refresh_policy_id': repr(self.refresh_policy_id), f'next_run_time': repr(
            self.next_run_time), f'cron': repr(self.cron), f'refresh_type': repr(self.refresh_type), f'error': repr(self.error)}
        class_name = "RefreshSchedule"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'refresh_policy_id': self.refresh_policy_id, 'next_run_time': self.next_run_time,
                'cron': self.cron, 'refresh_type': self.refresh_type, 'error': self.error}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
