from .return_class import AbstractApiClass


class StreamingRowCount(AbstractApiClass):
    """
        Returns the number of rows in a streaming feature group from the specified time

        Args:
            client (ApiClient): An authenticated API Client instance
            count (int): The number of rows in the feature group
            startTsMs (int): The start time for the number of rows.
    """

    def __init__(self, client, count=None, startTsMs=None):
        super().__init__(client, None)
        self.count = count
        self.start_ts_ms = startTsMs
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'count': repr(self.count),
                     f'start_ts_ms': repr(self.start_ts_ms)}
        class_name = "StreamingRowCount"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'count': self.count, 'start_ts_ms': self.start_ts_ms}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
