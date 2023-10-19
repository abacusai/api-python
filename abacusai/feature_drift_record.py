from .return_class import AbstractApiClass


class FeatureDriftRecord(AbstractApiClass):
    """
        Value of each type of drift

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): Name of feature.
            distance (float): Symmetric sum of KL divergences between the training distribution and the range of values in the specified window.
            jsDistance (float): JS divergence between the training distribution and the range of values in the specified window.
            wsDistance (float): Wasserstein distance between the training distribution and the range of values in the specified window.
            ksStatistic (float): Kolmogorov-Smirnov statistic computed between the training distribution and the range of values in the specified window.
    """

    def __init__(self, client, name=None, distance=None, jsDistance=None, wsDistance=None, ksStatistic=None):
        super().__init__(client, None)
        self.name = name
        self.distance = distance
        self.js_distance = jsDistance
        self.ws_distance = wsDistance
        self.ks_statistic = ksStatistic

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'distance': repr(self.distance), f'js_distance': repr(
            self.js_distance), f'ws_distance': repr(self.ws_distance), f'ks_statistic': repr(self.ks_statistic)}
        class_name = "FeatureDriftRecord"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'distance': self.distance, 'js_distance': self.js_distance,
                'ws_distance': self.ws_distance, 'ks_statistic': self.ks_statistic}
        return {key: value for key, value in resp.items() if value is not None}
