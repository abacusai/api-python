from .categorical_range_violation import CategoricalRangeViolation
from .null_violation import NullViolation
from .range_violation import RangeViolation
from .return_class import AbstractApiClass


class FeatureDriftSummary(AbstractApiClass):
    """
        Summary of important model monitoring statistics for features available in a model monitoring instance

        Args:
            client (ApiClient): An authenticated API Client instance
            featureIndex (list[dict]): A list of dicts of eligible feature names and corresponding overall feature drift measures.
            name (str): Name of feature.
            distance (float): Symmetric sum of KL divergences between the training distribution and the range of values in the specified window.
            jsDistance (float): JS divergence between the training distribution and the range of values in the specified window.
            wsDistance (float): Wasserstein distance between the training distribution and the range of values in the specified window.
            ksStatistic (float): Kolmogorov-Smirnov statistic computed between the training distribution and the range of values in the specified window.
            predictionDrift (float): Drift for the target column.
            targetColumn (str): Target column name.
            dataIntegrityTimeseries (dict): Frequency vs Data Integrity Violation Charts.
            nestedSummary (list[dict]): Summary of model monitoring statistics for nested features.
            psi (float): Population stability index computed between the training distribution and the range of values in the specified window.
            csi (float): Characteristic Stability Index computed between the training distribution and the range of values in the specified window.
            chiSquare (float): Chi-square statistic computed between the training distribution and the range of values in the specified window.
            nullViolations (NullViolation): A list of dicts of feature names and a description of corresponding null violations.
            rangeViolations (RangeViolation): A list of dicts of numerical feature names and corresponding prediction range discrepancies.
            catViolations (CategoricalRangeViolation): A list of dicts of categorical feature names and corresponding prediction range discrepancies.
    """

    def __init__(self, client, featureIndex=None, name=None, distance=None, jsDistance=None, wsDistance=None, ksStatistic=None, predictionDrift=None, targetColumn=None, dataIntegrityTimeseries=None, nestedSummary=None, psi=None, csi=None, chiSquare=None, nullViolations={}, rangeViolations={}, catViolations={}):
        super().__init__(client, None)
        self.feature_index = featureIndex
        self.name = name
        self.distance = distance
        self.js_distance = jsDistance
        self.ws_distance = wsDistance
        self.ks_statistic = ksStatistic
        self.prediction_drift = predictionDrift
        self.target_column = targetColumn
        self.data_integrity_timeseries = dataIntegrityTimeseries
        self.nested_summary = nestedSummary
        self.psi = psi
        self.csi = csi
        self.chi_square = chiSquare
        self.null_violations = client._build_class(
            NullViolation, nullViolations)
        self.range_violations = client._build_class(
            RangeViolation, rangeViolations)
        self.cat_violations = client._build_class(
            CategoricalRangeViolation, catViolations)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'feature_index': repr(self.feature_index), f'name': repr(self.name), f'distance': repr(self.distance), f'js_distance': repr(self.js_distance), f'ws_distance': repr(self.ws_distance), f'ks_statistic': repr(self.ks_statistic), f'prediction_drift': repr(self.prediction_drift), f'target_column': repr(
            self.target_column), f'data_integrity_timeseries': repr(self.data_integrity_timeseries), f'nested_summary': repr(self.nested_summary), f'psi': repr(self.psi), f'csi': repr(self.csi), f'chi_square': repr(self.chi_square), f'null_violations': repr(self.null_violations), f'range_violations': repr(self.range_violations), f'cat_violations': repr(self.cat_violations)}
        class_name = "FeatureDriftSummary"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'feature_index': self.feature_index, 'name': self.name, 'distance': self.distance, 'js_distance': self.js_distance, 'ws_distance': self.ws_distance, 'ks_statistic': self.ks_statistic, 'prediction_drift': self.prediction_drift, 'target_column': self.target_column, 'data_integrity_timeseries': self.data_integrity_timeseries,
                'nested_summary': self.nested_summary, 'psi': self.psi, 'csi': self.csi, 'chi_square': self.chi_square, 'null_violations': self._get_attribute_as_dict(self.null_violations), 'range_violations': self._get_attribute_as_dict(self.range_violations), 'cat_violations': self._get_attribute_as_dict(self.cat_violations)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
