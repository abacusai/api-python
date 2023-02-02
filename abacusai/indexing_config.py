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
        return f"IndexingConfig(primary_key={repr(self.primary_key)},\n  update_timestamp_key={repr(self.update_timestamp_key)},\n  lookup_keys={repr(self.lookup_keys)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'primary_key': self.primary_key, 'update_timestamp_key': self.update_timestamp_key, 'lookup_keys': self.lookup_keys}
