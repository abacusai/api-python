import dataclasses
from typing import List

from .abstract import ApiClass


@dataclasses.dataclass
class FeatureMappingConfig(ApiClass):
    feature_name: str
    feature_mapping: str = dataclasses.field(default=None)
    nested_feature_name: str = dataclasses.field(default=None)


@dataclasses.dataclass
class ProjectFeatureGroupTypeMappingsConfig(ApiClass):
    feature_group_id: str
    feature_group_type: str = dataclasses.field(default=None)
    feature_mappings: List[FeatureMappingConfig] = dataclasses.field(default=list)

    @classmethod
    def from_dict(cls, input_dict: dict):
        inst = cls(**input_dict)
        inst.feature_mappings = [FeatureMappingConfig.from_dict(fm) for fm in input_dict.get('feature_mappings') or []]
        return inst
