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
        repr_dict = {f'logs': repr(self.logs)}
        class_name = "DataPrepLogs"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'logs': self.logs}
        return {key: value for key, value in resp.items() if value is not None}
