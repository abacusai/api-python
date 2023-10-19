from .return_class import AbstractApiClass


class ProjectFeatureGroupSchemaVersion(AbstractApiClass):
    """
        A version of a schema

        Args:
            client (ApiClient): An authenticated API Client instance
            schemaVersion (id): The unique identifier of a schema version.
    """

    def __init__(self, client, schemaVersion=None):
        super().__init__(client, projectFeatureGroupSchemaVersion)
        self.schema_version = schemaVersion

    def __repr__(self):
        repr_dict = {f'schema_version': repr(self.schema_version)}
        class_name = "ProjectFeatureGroupSchemaVersion"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'schema_version': self.schema_version}
        return {key: value for key, value in resp.items() if value is not None}
