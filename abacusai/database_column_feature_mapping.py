from .return_class import AbstractApiClass


class DatabaseColumnFeatureMapping(AbstractApiClass):
    """
        Mapping for export of feature group version to database column

        Args:
            client (ApiClient): An authenticated API Client instance
            databaseColumn (str): database column name
            feature (str): feature group column it has been matched to
    """

    def __init__(self, client, databaseColumn=None, feature=None):
        super().__init__(client, None)
        self.database_column = databaseColumn
        self.feature = feature
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'database_column': repr(
            self.database_column), f'feature': repr(self.feature)}
        class_name = "DatabaseColumnFeatureMapping"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'database_column': self.database_column, 'feature': self.feature}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
