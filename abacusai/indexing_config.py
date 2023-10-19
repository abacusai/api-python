from .return_class import AbstractApiClass


class IndexingConfig(AbstractApiClass):
    """
        The indexing config for a Feature Group

        Args:
            client (ApiClient): An authenticated API Client instance
            primaryKey (str): A single key index
            updateTimestampKey (str): The primary timestamp feature
            lookupKeys (list[str]): A multi-key index. Cannot be used in conjuction with primary key.
    """

    def __init__(self, client, primaryKey=None, updateTimestampKey=None, lookupKeys=None):
        super().__init__(client, None)
        self.primary_key = primaryKey
        self.update_timestamp_key = updateTimestampKey
        self.lookup_keys = lookupKeys

    def __repr__(self):
        repr_dict = {f'primary_key': repr(self.primary_key), f'update_timestamp_key': repr(
            self.update_timestamp_key), f'lookup_keys': repr(self.lookup_keys)}
        class_name = "IndexingConfig"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'primary_key': self.primary_key, 'update_timestamp_key':
                self.update_timestamp_key, 'lookup_keys': self.lookup_keys}
        return {key: value for key, value in resp.items() if value is not None}
