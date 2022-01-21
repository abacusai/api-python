from .categorical_range_violation import CategoricalRangeViolation
from .null_violation import NullViolation
from .range_violation import RangeViolation
from .return_class import AbstractApiClass
from .type_violation import TypeViolation


class FeatureDriftSummary(AbstractApiClass):
    """
        Summary of important model monitoring statistics for features available in a model monitoring instance
    """

    def __init__(self, client, featureIndex=None, name=None, distance=None, jsDistance=None, predictionDrift=None, targetColumn=None, nullViolations={}, typeViolations={}, rangeViolations={}, catViolations={}):
        super().__init__(client, None)
        self.feature_index = featureIndex
        self.name = name
        self.distance = distance
        self.js_distance = jsDistance
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
        return f"FeatureDriftSummary(feature_index={repr(self.feature_index)},\n  name={repr(self.name)},\n  distance={repr(self.distance)},\n  js_distance={repr(self.js_distance)},\n  prediction_drift={repr(self.prediction_drift)},\n  target_column={repr(self.target_column)},\n  null_violations={repr(self.null_violations)},\n  type_violations={repr(self.type_violations)},\n  range_violations={repr(self.range_violations)},\n  cat_violations={repr(self.cat_violations)})"

    def to_dict(self):
        return {'feature_index': self.feature_index, 'name': self.name, 'distance': self.distance, 'js_distance': self.js_distance, 'prediction_drift': self.prediction_drift, 'target_column': self.target_column, 'null_violations': self._get_attribute_as_dict(self.null_violations), 'type_violations': self._get_attribute_as_dict(self.type_violations), 'range_violations': self._get_attribute_as_dict(self.range_violations), 'cat_violations': self._get_attribute_as_dict(self.cat_violations)}
