from .return_class import AbstractApiClass


class EmbeddingFeatureDriftDistribution(AbstractApiClass):
    """
        Feature distribution for embeddings

        Args:
            client (ApiClient): An authenticated API Client instance
            distance (list): Histogram data of KL divergences between the training distribution and the range of values in the specified window.
            jsDistance (list): Histogram data of JS divergence between the training distribution and the range of values in the specified window.
            wsDistance (list): Histogram data of Wasserstein distance between the training distribution and the range of values in the specified window.
            ksStatistic (list): Histogram data of Kolmogorov-Smirnov statistic computed between the training distribution and the range of values in the specified window.
            averageDrift (DriftTypesValue): Average drift embedding for each type of drift
    """

    def __init__(self, client, distance=None, jsDistance=None, wsDistance=None, ksStatistic=None, averageDrift={}):
        super().__init__(client, None)
        self.distance = distance
        self.js_distance = jsDistance
        self.ws_distance = wsDistance
        self.ks_statistic = ksStatistic
        self.average_drift = client._build_class(DriftTypesValue, averageDrift)

    def __repr__(self):
        repr_dict = {f'distance': repr(self.distance), f'js_distance': repr(self.js_distance), f'ws_distance': repr(
            self.ws_distance), f'ks_statistic': repr(self.ks_statistic), f'average_drift': repr(self.average_drift)}
        class_name = "EmbeddingFeatureDriftDistribution"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'distance': self.distance, 'js_distance': self.js_distance, 'ws_distance': self.ws_distance,
                'ks_statistic': self.ks_statistic, 'average_drift': self._get_attribute_as_dict(self.average_drift)}
        return {key: value for key, value in resp.items() if value is not None}
