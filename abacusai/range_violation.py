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

    def __repr__(self):
        return f"RangeViolation(name={repr(self.name)},\n  training_min={repr(self.training_min)},\n  training_max={repr(self.training_max)},\n  prediction_min={repr(self.prediction_min)},\n  prediction_max={repr(self.prediction_max)},\n  freq_above_training_range={repr(self.freq_above_training_range)},\n  freq_below_training_range={repr(self.freq_below_training_range)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'training_min': self.training_min, 'training_max': self.training_max, 'prediction_min': self.prediction_min, 'prediction_max': self.prediction_max, 'freq_above_training_range': self.freq_above_training_range, 'freq_below_training_range': self.freq_below_training_range}
