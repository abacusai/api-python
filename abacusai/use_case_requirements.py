from .return_class import AbstractApiClass


class UseCaseRequirements(AbstractApiClass):
    """
        Use Case Requirements

        Args:
            client (ApiClient): An authenticated API Client instance
            datasetType (str): The project-specific enum value of the dataset type
            name (str): The user-friendly name of the dataset type
            description (str): The description of the dataset type
            required (bool): True if the dataset type is required for this project
            allowedFeatureMappings (dict): A collection of key value pairs with each key being a column mapping enum (see a list of column mapping enums here) and each value being in the following dictionary format: { "description": str, "allowed_feature_types": feature_type_enum, "required": bool}
            allowedNestedFeatureMappings (dict): A collection of key value pairs with each key being a column mapping enum (see a list of column mapping enums here) and each value being in the following dictionary format: { "description": str, "allowed_feature_types": feature_type_enum, "required": bool}
    """

    def __init__(self, client, datasetType=None, name=None, description=None, required=None, allowedFeatureMappings=None, allowedNestedFeatureMappings=None):
        super().__init__(client, None)
        self.dataset_type = datasetType
        self.name = name
        self.description = description
        self.required = required
        self.allowed_feature_mappings = allowedFeatureMappings
        self.allowed_nested_feature_mappings = allowedNestedFeatureMappings

    def __repr__(self):
        return f"UseCaseRequirements(dataset_type={repr(self.dataset_type)},\n  name={repr(self.name)},\n  description={repr(self.description)},\n  required={repr(self.required)},\n  allowed_feature_mappings={repr(self.allowed_feature_mappings)},\n  allowed_nested_feature_mappings={repr(self.allowed_nested_feature_mappings)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'dataset_type': self.dataset_type, 'name': self.name, 'description': self.description, 'required': self.required, 'allowed_feature_mappings': self.allowed_feature_mappings, 'allowed_nested_feature_mappings': self.allowed_nested_feature_mappings}
