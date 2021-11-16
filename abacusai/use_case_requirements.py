from .return_class import AbstractApiClass


class UseCaseRequirements(AbstractApiClass):
    """
        Use Case Requirements
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
        return {'dataset_type': self.dataset_type, 'name': self.name, 'description': self.description, 'required': self.required, 'allowed_feature_mappings': self.allowed_feature_mappings, 'allowed_nested_feature_mappings': self.allowed_nested_feature_mappings}
