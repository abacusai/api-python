from .return_class import AbstractApiClass


class RoutingAction(AbstractApiClass):
    """
        Routing action

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name of the action.
            displayName (str): The display name of the action.
            value (str): The value of the action.
    """

    def __init__(self, client, name=None, displayName=None, value=None):
        super().__init__(client, None)
        self.name = name
        self.display_name = displayName
        self.value = value
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'display_name': repr(
            self.display_name), f'value': repr(self.value)}
        class_name = "RoutingAction"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name,
                'display_name': self.display_name, 'value': self.value}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
