from .daemon_task_event import DaemonTaskEvent
from .return_class import AbstractApiClass


class DaemonTaskEvents(AbstractApiClass):
    """
        The delivery log of an event-based daemon task.

        Args:
            client (ApiClient): An authenticated API Client instance
            totalCount (int): Total deliveries dispatched for the task (the list is capped).
            events (DaemonTaskEvent): Newest events first.
    """

    def __init__(self, client, totalCount=None, events={}):
        super().__init__(client, None)
        self.total_count = totalCount
        self.events = client._build_class(DaemonTaskEvent, events)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'total_count': repr(
            self.total_count), f'events': repr(self.events)}
        class_name = "DaemonTaskEvents"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'total_count': self.total_count,
                'events': self._get_attribute_as_dict(self.events)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
