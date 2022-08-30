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

    def __repr__(self):
        return f"ModelMonitorSummaryFromOrg(data={repr(self.data)},\n  infos={repr(self.infos)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'data': self.data, 'infos': self.infos}
