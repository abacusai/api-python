from .return_class import AbstractApiClass


class Vertical(AbstractApiClass):
    """
        A ChatLLM Vertical (e.g., Health).

        Args:
            client (ApiClient): An authenticated API Client instance
            externalApplicationId (str): The ID of the vertical's external application.
            name (str): The name of the vertical.
            description (str): The description of the vertical.
            verticalType (str): The type of vertical (e.g., HEALTH).
    """

    def __init__(self, client, externalApplicationId=None, name=None, description=None, verticalType=None):
        super().__init__(client, None)
        self.external_application_id = externalApplicationId
        self.name = name
        self.description = description
        self.vertical_type = verticalType
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'external_application_id': repr(self.external_application_id), f'name': repr(
            self.name), f'description': repr(self.description), f'vertical_type': repr(self.vertical_type)}
        class_name = "Vertical"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'external_application_id': self.external_application_id, 'name': self.name,
                'description': self.description, 'vertical_type': self.vertical_type}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
