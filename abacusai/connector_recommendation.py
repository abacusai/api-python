from .return_class import AbstractApiClass


class ConnectorRecommendation(AbstractApiClass):
    """
        One Ask-AI pick from the Connectors page's catalog, echoing the entry the page sent.

        Args:
            client (ApiClient): An authenticated API Client instance
            key (str): The page's catalog key for the entry (`app:<connector>` or `server:<gallery key>`).
            name (str): Display name.
            description (str): What it does.
    """

    def __init__(self, client, key=None, name=None, description=None):
        super().__init__(client, None)
        self.key = key
        self.name = name
        self.description = description
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'key': repr(self.key), f'name': repr(
            self.name), f'description': repr(self.description)}
        class_name = "ConnectorRecommendation"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'key': self.key, 'name': self.name,
                'description': self.description}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
