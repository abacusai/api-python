from .return_class import AbstractApiClass


class FeatureGroupRow(AbstractApiClass):
    """
        A row of a feature group.

        Args:
            client (ApiClient): An authenticated API Client instance
            featureGroupId (str): The ID of the feature group this row belongs to.
            primaryKey (str): Value of the primary key for this row.
            createdAt (str): The timestamp this feature group row was created in ISO-8601 format.
            updatedAt (str): The timestamp when this feature group row was last updated in ISO-8601 format.
            contents (dict): A dictionary of feature names and values for this row.
    """

    def __init__(self, client, featureGroupId=None, primaryKey=None, createdAt=None, updatedAt=None, contents=None):
        super().__init__(client, None)
        self.feature_group_id = featureGroupId
        self.primary_key = primaryKey
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.contents = contents
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'feature_group_id': repr(self.feature_group_id), f'primary_key': repr(self.primary_key), f'created_at': repr(
            self.created_at), f'updated_at': repr(self.updated_at), f'contents': repr(self.contents)}
        class_name = "FeatureGroupRow"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'feature_group_id': self.feature_group_id, 'primary_key': self.primary_key,
                'created_at': self.created_at, 'updated_at': self.updated_at, 'contents': self.contents}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
