

class UseCaseRequirements():
    '''
        Use Case Requirements
    '''

    def __init__(self, client, datasetType=None, name=None, description=None, required=None, allowedFeatureMappings=None):
        self.client = client
        self.id = None
        self.dataset_type = datasetType
        self.name = name
        self.description = description
        self.required = required
        self.allowed_feature_mappings = allowedFeatureMappings

    def __repr__(self):
        return f"UseCaseRequirements(dataset_type={repr(self.dataset_type)}, name={repr(self.name)}, description={repr(self.description)}, required={repr(self.required)}, allowed_feature_mappings={repr(self.allowed_feature_mappings)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'dataset_type': self.dataset_type, 'name': self.name, 'description': self.description, 'required': self.required, 'allowed_feature_mappings': self.allowed_feature_mappings}
