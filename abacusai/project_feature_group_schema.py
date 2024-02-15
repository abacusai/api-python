from .project_config import ProjectConfig
from .return_class import AbstractApiClass
from .schema import Schema


class ProjectFeatureGroupSchema(AbstractApiClass):
    """
        A schema description for a project feature group

        Args:
            client (ApiClient): An authenticated API Client instance
            nestedSchema (list): List of schema of nested features
            schema (Schema): List of schema description for the feature
            duplicateFeatures (Schema): List of duplicate featureschemas
            projectConfig (ProjectConfig): Project-specific config for this feature group.
    """

    def __init__(self, client, nestedSchema=None, schema={}, duplicateFeatures={}, projectConfig={}):
        super().__init__(client, None)
        self.nested_schema = nestedSchema
        self.schema = client._build_class(Schema, schema)
        self.duplicate_features = client._build_class(
            Schema, duplicateFeatures)
        self.project_config = client._build_class(ProjectConfig, projectConfig)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'nested_schema': repr(self.nested_schema), f'schema': repr(
            self.schema), f'duplicate_features': repr(self.duplicate_features), f'project_config': repr(self.project_config)}
        class_name = "ProjectFeatureGroupSchema"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'nested_schema': self.nested_schema, 'schema': self._get_attribute_as_dict(self.schema), 'duplicate_features': self._get_attribute_as_dict(
            self.duplicate_features), 'project_config': self._get_attribute_as_dict(self.project_config)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
