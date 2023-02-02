from .return_class import AbstractApiClass


class DataPrepLogs(AbstractApiClass):
    """
        Logs from data preparation.

        Args:
            client (ApiClient): An authenticated API Client instance
            logs (list[str]): List of logs from data preparation during model training.
    """

    def __init__(self, client, logs=None):
        super().__init__(client, None)
        self.logs = logs

    def __repr__(self):
        return f"DataPrepLogs(logs={repr(self.logs)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'logs': self.logs}
