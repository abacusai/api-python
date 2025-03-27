from .return_class import AbstractApiClass


class ComputePointInfo(AbstractApiClass):
    """
        The compute point info of the organization

        Args:
            client (ApiClient): An authenticated API Client instance
            updatedAt (str): The last time the compute point info was updated
            last24HoursUsage (int): The 24 hours usage of the organization
            last7DaysUsage (int): The 7 days usage of the organization
            currMonthAvailPoints (int): The current month's available compute points
            currMonthUsage (int): The current month's usage compute points
            lastThrottlePopUp (str): The last time the organization was throttled
            alwaysDisplay (bool): Whether to always display the compute point toggle
    """

    def __init__(self, client, updatedAt=None, last24HoursUsage=None, last7DaysUsage=None, currMonthAvailPoints=None, currMonthUsage=None, lastThrottlePopUp=None, alwaysDisplay=None):
        super().__init__(client, None)
        self.updated_at = updatedAt
        self.last_24_hours_usage = last24HoursUsage
        self.last_7_days_usage = last7DaysUsage
        self.curr_month_avail_points = currMonthAvailPoints
        self.curr_month_usage = currMonthUsage
        self.last_throttle_pop_up = lastThrottlePopUp
        self.always_display = alwaysDisplay
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'updated_at': repr(self.updated_at), f'last_24_hours_usage': repr(self.last_24_hours_usage), f'last_7_days_usage': repr(self.last_7_days_usage), f'curr_month_avail_points': repr(
            self.curr_month_avail_points), f'curr_month_usage': repr(self.curr_month_usage), f'last_throttle_pop_up': repr(self.last_throttle_pop_up), f'always_display': repr(self.always_display)}
        class_name = "ComputePointInfo"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'updated_at': self.updated_at, 'last_24_hours_usage': self.last_24_hours_usage, 'last_7_days_usage': self.last_7_days_usage, 'curr_month_avail_points':
                self.curr_month_avail_points, 'curr_month_usage': self.curr_month_usage, 'last_throttle_pop_up': self.last_throttle_pop_up, 'always_display': self.always_display}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
