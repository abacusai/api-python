import dataclasses
from typing import List

from .abstract import ApiClass


@dataclasses.dataclass
class FeatureMappingConfig(ApiClass):
    """
    Feature mapping configuration for a feature group type.

    Args:
        feature_name (str): The name of the feature in the feature group.
        feature_mapping (str): The desired feature mapping for the feature.
        nested_feature_name (str): The name of the nested feature in the feature group.
    """
    feature_name: str
    feature_mapping: str = dataclasses.field(default=None)
    nested_feature_name: str = dataclasses.field(default=None)


@dataclasses.dataclass
class ProjectFeatureGroupTypeMappingsConfig(ApiClass):
    """
    Project feature group type mappings.

    Args:
        feature_group_id (str): The unique identifier for the feature group.
        feature_group_type (str): The feature group type.
        feature_mappings (List[FeatureMappingConfig]): The feature mappings for the feature group.
    """
    feature_group_id: str
    feature_group_type: str = dataclasses.field(default=None)
    feature_mappings: List[FeatureMappingConfig] = dataclasses.field(default=list)

    @classmethod
    def from_dict(cls, input_dict: dict):
        inst = cls(**input_dict)
        inst.feature_mappings = [FeatureMappingConfig.from_dict(fm) for fm in input_dict.get('feature_mappings') or []]
        return inst
