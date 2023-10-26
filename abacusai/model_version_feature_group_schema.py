from .return_class import AbstractApiClass
from .schema import Schema


class ModelVersionFeatureGroupSchema(AbstractApiClass):
    """
        Schema for a feature group used in model version

        Args:
            client (ApiClient): An authenticated API Client instance
            featureGroupId (str): The ID of the feature group.
            featureGroupName (str): The name of the feature group.
            schema (Schema): List of feature schemas of a feature group.
    """

    def __init__(self, client, featureGroupId=None, featureGroupName=None, schema={}):
        super().__init__(client, None)
        self.feature_group_id = featureGroupId
        self.feature_group_name = featureGroupName
        self.schema = client._build_class(Schema, schema)

    def __repr__(self):
        repr_dict = {f'feature_group_id': repr(self.feature_group_id), f'feature_group_name': repr(
            self.feature_group_name), f'schema': repr(self.schema)}
        class_name = "ModelVersionFeatureGroupSchema"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'feature_group_id': self.feature_group_id, 'feature_group_name':
                self.feature_group_name, 'schema': self._get_attribute_as_dict(self.schema)}
        return {key: value for key, value in resp.items() if value is not None}
