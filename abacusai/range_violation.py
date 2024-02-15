from .return_class import AbstractApiClass


class RangeViolation(AbstractApiClass):
    """
        Summary of important range mismatches for a numerical feature discovered by a model monitoring instance

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): Name of feature.
            trainingMin (float): Minimum value of training distribution for the specified feature.
            trainingMax (float): Maximum value of training distribution for the specified feature.
            predictionMin (float): Minimum value of prediction distribution for the specified feature.
            predictionMax (float): Maximum value of prediction distribution for the specified feature.
            freqAboveTrainingRange (float): Frequency of prediction rows below training minimum for the specified feature.
            freqBelowTrainingRange (float): Frequency of prediction rows above training maximum for the specified feature.
    """

    def __init__(self, client, name=None, trainingMin=None, trainingMax=None, predictionMin=None, predictionMax=None, freqAboveTrainingRange=None, freqBelowTrainingRange=None):
        super().__init__(client, None)
        self.name = name
        self.training_min = trainingMin
        self.training_max = trainingMax
        self.prediction_min = predictionMin
        self.prediction_max = predictionMax
        self.freq_above_training_range = freqAboveTrainingRange
        self.freq_below_training_range = freqBelowTrainingRange
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'training_min': repr(self.training_min), f'training_max': repr(self.training_max), f'prediction_min': repr(self.prediction_min), f'prediction_max': repr(
            self.prediction_max), f'freq_above_training_range': repr(self.freq_above_training_range), f'freq_below_training_range': repr(self.freq_below_training_range)}
        class_name = "RangeViolation"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'training_min': self.training_min, 'training_max': self.training_max, 'prediction_min': self.prediction_min,
                'prediction_max': self.prediction_max, 'freq_above_training_range': self.freq_above_training_range, 'freq_below_training_range': self.freq_below_training_range}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
