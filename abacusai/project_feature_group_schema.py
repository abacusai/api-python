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

    def __repr__(self):
        return f"ProjectFeatureGroupSchema(nested_schema={repr(self.nested_schema)},\n  schema={repr(self.schema)},\n  duplicate_features={repr(self.duplicate_features)},\n  project_config={repr(self.project_config)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'nested_schema': self.nested_schema, 'schema': self._get_attribute_as_dict(self.schema), 'duplicate_features': self._get_attribute_as_dict(self.duplicate_features), 'project_config': self._get_attribute_as_dict(self.project_config)}
