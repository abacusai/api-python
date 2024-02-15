from .return_class import AbstractApiClass


class NullViolation(AbstractApiClass):
    """
        Summary of anomalous null frequencies for a feature discovered by a model monitoring instance

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): Name of feature.
            violation (str): Description of null violation for a prediction feature.
            trainingNullFreq (float): Proportion of null entries in training feature.
            predictionNullFreq (float): Proportion of null entries in prediction feature.
    """

    def __init__(self, client, name=None, violation=None, trainingNullFreq=None, predictionNullFreq=None):
        super().__init__(client, None)
        self.name = name
        self.violation = violation
        self.training_null_freq = trainingNullFreq
        self.prediction_null_freq = predictionNullFreq
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'violation': repr(self.violation), f'training_null_freq': repr(
            self.training_null_freq), f'prediction_null_freq': repr(self.prediction_null_freq)}
        class_name = "NullViolation"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'violation': self.violation, 'training_null_freq':
                self.training_null_freq, 'prediction_null_freq': self.prediction_null_freq}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
