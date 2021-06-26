from .feature_group import FeatureGroup


class FeatureGroupVersion():
    '''

    '''

    def __init__(self, client, featureGroupVersion=None, createdAt=None, updatedAt=None, instanceInfo=None, schemaValues=None, lifecycle=None, featureGroup={}):
        self.client = client
        self.id = featureGroupVersion
        self.feature_group_version = featureGroupVersion
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.instance_info = instanceInfo
        self.schema_values = schemaValues
        self.lifecycle = lifecycle
        self.feature_group = client._build_class(FeatureGroup, featureGroup)

    def __repr__(self):
        return f"FeatureGroupVersion(feature_group_version={repr(self.feature_group_version)}, created_at={repr(self.created_at)}, updated_at={repr(self.updated_at)}, instance_info={repr(self.instance_info)}, schema_values={repr(self.schema_values)}, lifecycle={repr(self.lifecycle)}, feature_group={repr(self.feature_group)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'feature_group_version': self.feature_group_version, 'created_at': self.created_at, 'updated_at': self.updated_at, 'instance_info': self.instance_info, 'schema_values': self.schema_values, 'lifecycle': self.lifecycle, 'feature_group': [elem.to_dict() for elem in self.feature_group or []]}
