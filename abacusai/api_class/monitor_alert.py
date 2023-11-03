import dataclasses
from typing import List

from . import enums
from .abstract import ApiClass, _ApiClassFactory


@dataclasses.dataclass
class AlertConditionConfig(ApiClass):
    alert_type: enums.MonitorAlertType = dataclasses.field(default=None, init=True)

    @classmethod
    def _get_builder(cls):
        return _AlertConditionConfigFactory


@dataclasses.dataclass
class AccuracyBelowThresholdConditionConfig(AlertConditionConfig):
    """
    Accuracy Below Threshold Condition Config for Monitor Alerts
    Args:
        threshold (float): Threshold for when to consider a column to be in violation. The alert will only fire when the drift value is strictly greater than the threshold.
    """
    threshold: float = dataclasses.field(default=None)

    def __post_init__(self):
        self.alert_type = enums.MonitorAlertType.ACCURACY_BELOW_THRESHOLD


@dataclasses.dataclass
class FeatureDriftConditionConfig(AlertConditionConfig):
    """
    Feature Drift Condition Config for Monitor Alerts
    Args:
        feature_drift_type (FeatureDriftType): Feature drift type to apply the threshold on to determine whether a column has drifted significantly enough to be a violation.
        threshold (float): Threshold for when to consider a column to be in violation. The alert will only fire when the drift value is strictly greater than the threshold.
        minimum_violations (int): Number of columns that must exceed the specified threshold to trigger an alert.
    """
    feature_drift_type: enums.FeatureDriftType = dataclasses.field(default=None)
    threshold: float = dataclasses.field(default=None)
    minimum_violations: int = dataclasses.field(default=None)

    def __post_init__(self):
        self.alert_type = enums.MonitorAlertType.FEATURE_DRIFT


@dataclasses.dataclass
class DataIntegrityViolationConditionConfig(AlertConditionConfig):
    """
    Data Integrity Violation Condition Config for Monitor Alerts
    Args:
        data_integrity_type (DataIntegrityViolationType): This option selects the data integrity violations to monitor for this alert.
        minimum_violations (int): Number of columns that must exceed the specified threshold to trigger an alert.
    """
    data_integrity_type: enums.DataIntegrityViolationType = dataclasses.field(default=None)
    minimum_violations: int = dataclasses.field(default=None)

    def __post_init__(self):
        self.alert_type = enums.MonitorAlertType.DATA_INTEGRITY_VIOLATIONS


@dataclasses.dataclass
class BiasViolationConditionConfig(AlertConditionConfig):
    """
    Bias Violation Condition Config for Monitor Alerts
    Args:
        bias_type (BiasType): This option selects the bias metric to monitor for this alert.
        threshold (float): Threshold for when to consider a column to be in violation. The alert will only fire when the drift value is strictly greater than the threshold.
        minimum_violations (int): Number of columns that must exceed the specified threshold to trigger an alert.
    """
    bias_type: enums.BiasType = dataclasses.field(default=None)
    threshold: float = dataclasses.field(default=None)
    minimum_violations: int = dataclasses.field(default=None)

    def __post_init__(self):
        self.alert_type = enums.MonitorAlertType.BIAS_VIOLATIONS


@dataclasses.dataclass
class _AlertConditionConfigFactory(_ApiClassFactory):
    config_abstract_class = AlertConditionConfig
    config_class_key = 'alert_type'
    config_class_key_value_camel_case = True
    config_class_map = {
        enums.MonitorAlertType.ACCURACY_BELOW_THRESHOLD: AccuracyBelowThresholdConditionConfig,
        enums.MonitorAlertType.FEATURE_DRIFT: FeatureDriftConditionConfig,
        enums.MonitorAlertType.DATA_INTEGRITY_VIOLATIONS: DataIntegrityViolationConditionConfig,
        enums.MonitorAlertType.BIAS_VIOLATIONS: BiasViolationConditionConfig,
    }


@dataclasses.dataclass
class AlertActionConfig(ApiClass):
    action_type: enums.AlertActionType = dataclasses.field(default=None, repr=False, init=True)

    @classmethod
    def _get_builder(cls):
        return _AlertActionConfigFactory


@dataclasses.dataclass
class EmailActionConfig(AlertActionConfig):
    """
    Email Action Config for Monitor Alerts
    Args:
        email_recipients (List[str]): List of email addresses to send the alert to.
        email_body (str): Body of the email to send.
    """
    email_recipients: List[str] = dataclasses.field(default=None)
    email_body: str = dataclasses.field(default=None)

    def __post_init__(self):
        self.action_type = enums.AlertActionType.EMAIL


@dataclasses.dataclass
class _AlertActionConfigFactory(_ApiClassFactory):
    config_abstract_class = AlertActionConfig
    config_class_key = 'action_type'
    config_class_map = {
        enums.AlertActionType.EMAIL: EmailActionConfig,
    }
