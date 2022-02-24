from .categorical_range_violation import CategoricalRangeViolation
from .null_violation import NullViolation
from .range_violation import RangeViolation
from .return_class import AbstractApiClass
from .type_violation import TypeViolation


class FeatureDriftSummary(AbstractApiClass):
    """
        Summary of important model monitoring statistics for features available in a model monitoring instance

        Args:
            client (ApiClient): An authenticated API Client instance
            featureIndex (list of json objects): A list of dicts of eligible feature names and corresponding overall feature drift measures.
            name (str): Name of feature.
            distance (float): Symmetric sum of KL divergences between the training distribution and the range of values in the specified window.
            jsDistance (float): JS divergence between the training distribution and the range of values in the specified window.
            wsDistance (float): Wasserstein distance between the training distribution and the range of values in the specified window.
            ksStatistic (float): Kolmogorov-Smirnov statistic computed between the training distribution and the range of values in the specified window.
            predictionDrift (float): Drift for the target column.
            targetColumn (str): Target column name.
            nullViolations (NullViolation): A list of dicts of feature names and a description of corresponding null violations.
            typeViolations (TypeViolation): A list of dicts of feature names and corresponding type mismatches.
            rangeViolations (RangeViolation): A list of dicts of numerical feature names and corresponding prediction range discrepancies.
            catViolations (CategoricalRangeViolation): A list of dicts of categorical feature names and corresponding prediction range discrepancies.
    """

    def __init__(self, client, featureIndex=None, name=None, distance=None, jsDistance=None, wsDistance=None, ksStatistic=None, predictionDrift=None, targetColumn=None, nullViolations={}, typeViolations={}, rangeViolations={}, catViolations={}):
        super().__init__(client, None)
        self.feature_index = featureIndex
        self.name = name
        self.distance = distance
        self.js_distance = jsDistance
        self.ws_distance = wsDistance
        self.ks_statistic = ksStatistic
        self.prediction_drift = predictionDrift
        self.target_column = targetColumn
        self.null_violations = client._build_class(
            NullViolation, nullViolations)
        self.type_violations = client._build_class(
            TypeViolation, typeViolations)
        self.range_violations = client._build_class(
            RangeViolation, rangeViolations)
        self.cat_violations = client._build_class(
            CategoricalRangeViolation, catViolations)

    def __repr__(self):
        return f"FeatureDriftSummary(feature_index={repr(self.feature_index)},\n  name={repr(self.name)},\n  distance={repr(self.distance)},\n  js_distance={repr(self.js_distance)},\n  ws_distance={repr(self.ws_distance)},\n  ks_statistic={repr(self.ks_statistic)},\n  prediction_drift={repr(self.prediction_drift)},\n  target_column={repr(self.target_column)},\n  null_violations={repr(self.null_violations)},\n  type_violations={repr(self.type_violations)},\n  range_violations={repr(self.range_violations)},\n  cat_violations={repr(self.cat_violations)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'feature_index': self.feature_index, 'name': self.name, 'distance': self.distance, 'js_distance': self.js_distance, 'ws_distance': self.ws_distance, 'ks_statistic': self.ks_statistic, 'prediction_drift': self.prediction_drift, 'target_column': self.target_column, 'null_violations': self._get_attribute_as_dict(self.null_violations), 'type_violations': self._get_attribute_as_dict(self.type_violations), 'range_violations': self._get_attribute_as_dict(self.range_violations), 'cat_violations': self._get_attribute_as_dict(self.cat_violations)}
