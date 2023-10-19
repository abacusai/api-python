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
        repr_dict = {f'data': repr(self.data)}
        class_name = "FeatureRecord"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'data': self.data}
        return {key: value for key, value in resp.items() if value is not None}
