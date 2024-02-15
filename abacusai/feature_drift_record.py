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
            psi (float): Population stability index computed between the training distribution and the range of values in the specified window.
            csi (float): Characteristic Stability Index computed between the training distribution and the range of values in the specified window.
            chiSquare (float): Chi-square statistic computed between the training distribution and the range of values in the specified window.
    """

    def __init__(self, client, name=None, distance=None, jsDistance=None, wsDistance=None, ksStatistic=None, psi=None, csi=None, chiSquare=None):
        super().__init__(client, None)
        self.name = name
        self.distance = distance
        self.js_distance = jsDistance
        self.ws_distance = wsDistance
        self.ks_statistic = ksStatistic
        self.psi = psi
        self.csi = csi
        self.chi_square = chiSquare
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'distance': repr(self.distance), f'js_distance': repr(self.js_distance), f'ws_distance': repr(
            self.ws_distance), f'ks_statistic': repr(self.ks_statistic), f'psi': repr(self.psi), f'csi': repr(self.csi), f'chi_square': repr(self.chi_square)}
        class_name = "FeatureDriftRecord"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'distance': self.distance, 'js_distance': self.js_distance, 'ws_distance': self.ws_distance,
                'ks_statistic': self.ks_statistic, 'psi': self.psi, 'csi': self.csi, 'chi_square': self.chi_square}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
