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

    def __repr__(self):
        return f"FeatureGroupRow(feature_group_id={repr(self.feature_group_id)},\n  primary_key={repr(self.primary_key)},\n  created_at={repr(self.created_at)},\n  updated_at={repr(self.updated_at)},\n  contents={repr(self.contents)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'feature_group_id': self.feature_group_id, 'primary_key': self.primary_key, 'created_at': self.created_at, 'updated_at': self.updated_at, 'contents': self.contents}
