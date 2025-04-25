from .return_class import AbstractApiClass


class HostedDatabase(AbstractApiClass):
    """
        Hosted Database

        Args:
            client (ApiClient): An authenticated API Client instance
            hostedDatabaseId (id): The ID of the hosted database
            displayName (str): The name of the hosted database
            createdAt (str): The creation timestamp
            updatedAt (str): The last update timestamp
            lifecycle (str): The lifecycle of the hosted database
    """

    def __init__(self, client, hostedDatabaseId=None, displayName=None, createdAt=None, updatedAt=None, lifecycle=None):
        super().__init__(client, hostedDatabaseId)
        self.hosted_database_id = hostedDatabaseId
        self.display_name = displayName
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.lifecycle = lifecycle
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'hosted_database_id': repr(self.hosted_database_id), f'display_name': repr(self.display_name), f'created_at': repr(
            self.created_at), f'updated_at': repr(self.updated_at), f'lifecycle': repr(self.lifecycle)}
        class_name = "HostedDatabase"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'hosted_database_id': self.hosted_database_id, 'display_name': self.display_name,
                'created_at': self.created_at, 'updated_at': self.updated_at, 'lifecycle': self.lifecycle}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
