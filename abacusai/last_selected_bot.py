from .return_class import AbstractApiClass


class LastSelectedBot(AbstractApiClass):
    """
        The last selected bot for the user in the organization.

        Args:
            client (ApiClient): An authenticated API Client instance
            externalApplicationId (str): The ID of the last selected bot (External Application).
            mode (str): The UI mode selection. One of 'chat', 'auto', 'agent', 'custom_bot'.
            createdAt (str): Timestamp when the record was first created.
            updatedAt (str): Timestamp when the record was last modified.
    """

    def __init__(self, client, externalApplicationId=None, mode=None, createdAt=None, updatedAt=None):
        super().__init__(client, None)
        self.external_application_id = externalApplicationId
        self.mode = mode
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'external_application_id': repr(self.external_application_id), f'mode': repr(
            self.mode), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at)}
        class_name = "LastSelectedBot"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'external_application_id': self.external_application_id,
                'mode': self.mode, 'created_at': self.created_at, 'updated_at': self.updated_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
