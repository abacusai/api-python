from .return_class import AbstractApiClass


class BatchPredictionVersionLogs(AbstractApiClass):
    """
        Logs from batch prediction version.

        Args:
            client (ApiClient): An authenticated API Client instance
            logs (list[str]): List of logs from batch prediction version.
            warnings (list[str]): List of warnings from batch prediction version.
    """

    def __init__(self, client, logs=None, warnings=None):
        super().__init__(client, None)
        self.logs = logs
        self.warnings = warnings

    def __repr__(self):
        repr_dict = {f'logs': repr(
            self.logs), f'warnings': repr(self.warnings)}
        class_name = "BatchPredictionVersionLogs"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'logs': self.logs, 'warnings': self.warnings}
        return {key: value for key, value in resp.items() if value is not None}
