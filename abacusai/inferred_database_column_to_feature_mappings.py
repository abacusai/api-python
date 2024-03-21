from .database_column_feature_mapping import DatabaseColumnFeatureMapping
from .return_class import AbstractApiClass


class InferredDatabaseColumnToFeatureMappings(AbstractApiClass):
    """
        Autocomplete mappings for database to connector columns

        Args:
            client (ApiClient): An authenticated API Client instance
            databaseColumnToFeatureMappings (DatabaseColumnFeatureMapping): Database columns feature mappings
    """

    def __init__(self, client, databaseColumnToFeatureMappings={}):
        super().__init__(client, None)
        self.database_column_to_feature_mappings = client._build_class(
            DatabaseColumnFeatureMapping, databaseColumnToFeatureMappings)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'database_column_to_feature_mappings': repr(
            self.database_column_to_feature_mappings)}
        class_name = "InferredDatabaseColumnToFeatureMappings"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'database_column_to_feature_mappings': self._get_attribute_as_dict(
            self.database_column_to_feature_mappings)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
