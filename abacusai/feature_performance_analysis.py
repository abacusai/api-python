from .return_class import AbstractApiClass


class FeaturePerformanceAnalysis(AbstractApiClass):
    """
        A feature performance analysis for Monitor

        Args:
            client (ApiClient): An authenticated API Client instance
            features (list): A list of the features that are being analyzed.
            featureMetrics (list): A list of dictionary for every feature and its metrics
            metricsKeys (list): A list of the keys for the metrics.
    """

    def __init__(self, client, features=None, featureMetrics=None, metricsKeys=None):
        super().__init__(client, None)
        self.features = features
        self.feature_metrics = featureMetrics
        self.metrics_keys = metricsKeys
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'features': repr(self.features), f'feature_metrics': repr(
            self.feature_metrics), f'metrics_keys': repr(self.metrics_keys)}
        class_name = "FeaturePerformanceAnalysis"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'features': self.features, 'feature_metrics': self.feature_metrics,
                'metrics_keys': self.metrics_keys}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
