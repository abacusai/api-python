import dataclasses
from typing import List, Optional

from . import enums
from .abstract import ApiClass, _ApiClassFactory


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


@dataclasses.dataclass
class ConstraintConfig(ApiClass):
    """
    Constraint configuration.

    Args:
        constant (float): The constant value for the constraint.
        operator (str): The operator for the constraint. Could be 'EQ', 'LE', 'GE'
        enforcement (str): The enforcement for the constraint. Could be 'HARD' or 'SOFT' or 'SKIP'. Default is 'HARD'
        code (str): The code for the constraint.
        penalty (float): The penalty for violating the constraint.
    """
    constant: float
    operator: str
    enforcement: Optional[str] = dataclasses.field(default=None)
    code: Optional[str] = dataclasses.field(default=None)
    penalty: Optional[float] = dataclasses.field(default=None)


@dataclasses.dataclass
class ProjectFeatureGroupConfig(ApiClass):
    """
    An abstract class for project feature group configuration.
    """
    type: enums.ProjectConfigType = dataclasses.field(default=None, repr=False, init=False)

    @classmethod
    def _get_builder(cls):
        return _ProjectFeatureGroupConfigFactory


@dataclasses.dataclass
class ConstraintProjectFeatureGroupConfig(ProjectFeatureGroupConfig):
    """
    Constraint project feature group configuration.

    Args:
        constraints (List[ConstraintConfig]): The constraint for the feature group. Should be a list of one ConstraintConfig.
    """
    constraints: List[ConstraintConfig]

    def __post_init__(self):
        self.type = enums.ProjectConfigType.CONSTRAINTS


@dataclasses.dataclass
class ReviewModeProjectFeatureGroupConfig(ProjectFeatureGroupConfig):
    """
    Review mode project feature group configuration.

    Args:
        is_review_mode (bool): The review mode for the feature group.
    """
    is_review_mode: bool

    def __post_init__(self):
        self.type = enums.ProjectConfigType.REVIEW_MODE


@dataclasses.dataclass
class _ProjectFeatureGroupConfigFactory(_ApiClassFactory):
    config_abstract_class = ProjectFeatureGroupConfig
    config_class_key = 'type'
    config_class_map = {
        enums.ProjectConfigType.CONSTRAINTS: ConstraintProjectFeatureGroupConfig,
        enums.ProjectConfigType.REVIEW_MODE: ReviewModeProjectFeatureGroupConfig
    }
