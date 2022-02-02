from .return_class import AbstractApiClass


class FeatureRecord(AbstractApiClass):
    """
        A feature record

        Args:
            client (ApiClient): An authenticated API Client instance
            data (dict): the record's current data
    """

    def __init__(self, client, data=None):
        super().__init__(client, None)
        self.data = data

    def __repr__(self):
        return f"FeatureRecord(data={repr(self.data)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'data': self.data}
