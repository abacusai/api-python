from .return_class import AbstractApiClass


class ModelMonitorSummaryFromOrg(AbstractApiClass):
    """
        A summary of model monitor given an organization

        Args:
            client (ApiClient): An authenticated API Client instance
            data (list): A list of either model accuracy, drift, data integrity, or bias chart objects and their monitor version information.
            infos (dict): A dictionary of model monitor information.
    """

    def __init__(self, client, data=None, infos=None):
        super().__init__(client, None)
        self.data = data
        self.infos = infos
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'data': repr(self.data), f'infos': repr(self.infos)}
        class_name = "ModelMonitorSummaryFromOrg"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'data': self.data, 'infos': self.infos}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
