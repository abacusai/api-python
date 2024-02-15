from .embedding_feature_drift_distribution import EmbeddingFeatureDriftDistribution
from .forecasting_monitor_summary import ForecastingMonitorSummary
from .return_class import AbstractApiClass


class MonitorDriftAndDistributions(AbstractApiClass):
    """
        Summary of important model monitoring statistics for features available in a model monitoring instance

        Args:
            client (ApiClient): An authenticated API Client instance
            featureDrifts (list[dict]): A list of dicts of eligible feature names and corresponding overall feature drift measures.
            featureDistributions (list[dict]): A list of dicts of feature names and corresponding feature distributions.
            nestedDrifts (list[dict]): A list of dicts of nested feature names and corresponding overall feature drift measures.
            forecastingMonitorSummary (ForecastingMonitorSummary): Summary of important model monitoring statistics for features available in a model monitoring instance
            embeddingsDistribution (EmbeddingFeatureDriftDistribution): Summary of important model monitoring statistics for features available in a model monitoring instance
    """

    def __init__(self, client, featureDrifts=None, featureDistributions=None, nestedDrifts=None, forecastingMonitorSummary={}, embeddingsDistribution={}):
        super().__init__(client, None)
        self.feature_drifts = featureDrifts
        self.feature_distributions = featureDistributions
        self.nested_drifts = nestedDrifts
        self.forecasting_monitor_summary = client._build_class(
            ForecastingMonitorSummary, forecastingMonitorSummary)
        self.embeddings_distribution = client._build_class(
            EmbeddingFeatureDriftDistribution, embeddingsDistribution)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'feature_drifts': repr(self.feature_drifts), f'feature_distributions': repr(self.feature_distributions), f'nested_drifts': repr(
            self.nested_drifts), f'forecasting_monitor_summary': repr(self.forecasting_monitor_summary), f'embeddings_distribution': repr(self.embeddings_distribution)}
        class_name = "MonitorDriftAndDistributions"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'feature_drifts': self.feature_drifts, 'feature_distributions': self.feature_distributions, 'nested_drifts': self.nested_drifts,
                'forecasting_monitor_summary': self._get_attribute_as_dict(self.forecasting_monitor_summary), 'embeddings_distribution': self._get_attribute_as_dict(self.embeddings_distribution)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
