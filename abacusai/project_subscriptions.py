from .project_subscription_channel import ProjectSubscriptionChannel
from .project_subscription_schedule import ProjectSubscriptionSchedule
from .return_class import AbstractApiClass


class ProjectSubscriptions(AbstractApiClass):
    """
        Everything that wakes an AGENT_COORDINATOR project's coordinator (the project's Subscriptions card)

        Args:
            client (ApiClient): An authenticated API Client instance
            schedules (ProjectSubscriptionSchedule): Heartbeat and named tasks with their latest run.
            channels (ProjectSubscriptionChannel): GitHub / webhook event channels.
    """

    def __init__(self, client, schedules={}, channels={}):
        super().__init__(client, None)
        self.schedules = client._build_class(
            ProjectSubscriptionSchedule, schedules)
        self.channels = client._build_class(
            ProjectSubscriptionChannel, channels)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'schedules': repr(
            self.schedules), f'channels': repr(self.channels)}
        class_name = "ProjectSubscriptions"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'schedules': self._get_attribute_as_dict(
            self.schedules), 'channels': self._get_attribute_as_dict(self.channels)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
