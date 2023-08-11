from .return_class import AbstractApiClass


class DatasetVersionLogs(AbstractApiClass):
    """
        Logs from dataset version.

        Args:
            client (ApiClient): An authenticated API Client instance
            logs (list[str]): List of logs from dataset version.
    """

    def __init__(self, client, logs=None):
        super().__init__(client, None)
        self.logs = logs

    def __repr__(self):
        return f"DatasetVersionLogs(logs={repr(self.logs)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'logs': self.logs}
