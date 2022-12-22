from .return_class import AbstractApiClass


class EmbeddingFeatureDriftDistribution(AbstractApiClass):
    """
        distance (List): Histogram data of KL divergences between the training distribution and the range of values in the specified window.

        Args:
            client (ApiClient): An authenticated API Client instance
            distance (list): 
            jsDistance (list): 
            wsDistance (list): 
            ksStatistic (list): 
    """

    def __init__(self, client, distance=None, jsDistance=None, wsDistance=None, ksStatistic=None):
        super().__init__(client, None)
        self.distance = distance
        self.js_distance = jsDistance
        self.ws_distance = wsDistance
        self.ks_statistic = ksStatistic

    def __repr__(self):
        return f"EmbeddingFeatureDriftDistribution(distance={repr(self.distance)},\n  js_distance={repr(self.js_distance)},\n  ws_distance={repr(self.ws_distance)},\n  ks_statistic={repr(self.ks_statistic)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'distance': self.distance, 'js_distance': self.js_distance, 'ws_distance': self.ws_distance, 'ks_statistic': self.ks_statistic}
