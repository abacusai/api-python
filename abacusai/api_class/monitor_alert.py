import dataclasses
from typing import List

from . import enums
from .abstract import ApiClass, _ApiClassFactory


@dataclasses.dataclass
class AlertConditionConfig(ApiClass):
    """
    An abstract class for alert condition configs
    """
    alert_type: enums.MonitorAlertType = dataclasses.field(default=None, repr=False, init=False)

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
        feature_names (List[str]): List of feature names to monitor for this alert.
    """
    feature_drift_type: enums.FeatureDriftType = dataclasses.field(default=None)
    threshold: float = dataclasses.field(default=None)
    minimum_violations: int = dataclasses.field(default=None)
    feature_names: List[str] = dataclasses.field(default=None)

    def __post_init__(self):
        self.alert_type = enums.MonitorAlertType.FEATURE_DRIFT


@dataclasses.dataclass
class TargetDriftConditionConfig(AlertConditionConfig):
    """
    Target Drift Condition Config for Monitor Alerts

    Args:
        feature_drift_type (FeatureDriftType): Target drift type to apply the threshold on to determine whether a column has drifted significantly enough to be a violation.
        threshold (float): Threshold for when to consider the target column to be in violation. The alert will only fire when the drift value is strictly greater than the threshold.
    """
    feature_drift_type: enums.FeatureDriftType = dataclasses.field(default=None)
    threshold: float = dataclasses.field(default=None)

    def __post_init__(self):
        self.alert_type = enums.MonitorAlertType.TARGET_DRIFT


@dataclasses.dataclass
class HistoryLengthDriftConditionConfig(AlertConditionConfig):
    """
    History Length Drift Condition Config for Monitor Alerts

    Args:
        feature_drift_type (FeatureDriftType): History length drift type to apply the threshold on to determine whether the history length has drifted significantly enough to be a violation.
        threshold (float): Threshold for when to consider the history length  to be in violation. The alert will only fire when the drift value is strictly greater than the threshold.
    """
    feature_drift_type: enums.FeatureDriftType = dataclasses.field(default=None)
    threshold: float = dataclasses.field(default=None)

    def __post_init__(self):
        self.alert_type = enums.MonitorAlertType.HISTORY_LENGTH_DRIFT


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
class PredictionCountConditionConfig(AlertConditionConfig):
    """
    Deployment Prediction Condition Config for Deployment Alerts. By default we monitor if predictions made over a time window has reduced significantly.
    Args:
        threshold (float): Threshold for when to consider to be a violation. Negative means alert on reduction, positive means alert on increase.
        aggregation_window (str): Time window to aggregate the predictions over, e.g. 1h, 10m. Only h(hour), m(minute) and s(second) are supported.
        aggregation_type (str): Aggregation type to use for the aggregation window, e.g. sum, avg.
    """
    threshold: float = dataclasses.field(default=None)
    aggregation_window: str = dataclasses.field(default=None)
    aggregation_type: str = dataclasses.field(default=None)

    def __post_init__(self):
        self.alert_type = enums.MonitorAlertType.PREDICTION_COUNT


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
        enums.MonitorAlertType.TARGET_DRIFT: TargetDriftConditionConfig,
        enums.MonitorAlertType.HISTORY_LENGTH_DRIFT: HistoryLengthDriftConditionConfig,
        enums.MonitorAlertType.PREDICTION_COUNT: PredictionCountConditionConfig,
    }


@dataclasses.dataclass
class AlertActionConfig(ApiClass):
    """
    An abstract class for alert action configs
    """
    action_type: enums.AlertActionType = dataclasses.field(default=None, repr=False, init=False)

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


@dataclasses.dataclass
class MonitorThresholdConfig(ApiClass):
    """
    Monitor Threshold Config for Monitor Alerts

    Args:
        drift_type (FeatureDriftType): Feature drift type to apply the threshold on to determine whether a column has drifted significantly enough to be a violation.
        threshold_config (ThresholdConfigs): Thresholds for when to consider a column to be in violation. The alert will only fire when the drift value is strictly greater than the threshold.
    """
    drift_type: enums.FeatureDriftType = dataclasses.field(default=None)
    at_risk_threshold: float = dataclasses.field(default=None)
    severely_drifting_threshold: float = dataclasses.field(default=None)

    def to_dict(self):
        return {
            'drift_type': self.drift_type,
            'at_risk_threshold': self.at_risk_threshold,
            'severely_drifting_threshold': self.severely_drifting_threshold,
        }
