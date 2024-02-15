from .return_class import AbstractApiClass


class UseCaseRequirements(AbstractApiClass):
    """
        Use Case Requirements

        Args:
            client (ApiClient): An authenticated API Client instance
            datasetType (str): The project-specific enum value of the dataset type.
            name (str): The user-friendly name of the dataset type.
            description (str): The description of the dataset type.
            required (bool): True if the dataset type is required for this project.
            multi (bool): If true, multiple versions of the dataset type can be used for training.
            allowedFeatureMappings (dict): A collection of key-value pairs, with each key being a column mapping enum (see a list of column mapping enums here) and each value being in the following dictionary format: { "description": str, "allowed_feature_types": feature_type_enum, "required": bool }.
            allowedNestedFeatureMappings (dict): A collection of key-value pairs, with each key being a column mapping enum (see a list of column mapping enums here) and each value being in the following dictionary format: { "description": str, "allowed_feature_types": feature_type_enum, "required": bool }.
    """

    def __init__(self, client, datasetType=None, name=None, description=None, required=None, multi=None, allowedFeatureMappings=None, allowedNestedFeatureMappings=None):
        super().__init__(client, None)
        self.dataset_type = datasetType
        self.name = name
        self.description = description
        self.required = required
        self.multi = multi
        self.allowed_feature_mappings = allowedFeatureMappings
        self.allowed_nested_feature_mappings = allowedNestedFeatureMappings
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'dataset_type': repr(self.dataset_type), f'name': repr(self.name), f'description': repr(self.description), f'required': repr(self.required), f'multi': repr(
            self.multi), f'allowed_feature_mappings': repr(self.allowed_feature_mappings), f'allowed_nested_feature_mappings': repr(self.allowed_nested_feature_mappings)}
        class_name = "UseCaseRequirements"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'dataset_type': self.dataset_type, 'name': self.name, 'description': self.description, 'required': self.required, 'multi': self.multi,
                'allowed_feature_mappings': self.allowed_feature_mappings, 'allowed_nested_feature_mappings': self.allowed_nested_feature_mappings}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
