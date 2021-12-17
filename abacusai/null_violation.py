from .return_class import AbstractApiClass


class NullViolation(AbstractApiClass):
    """
        Summary of anomalous null frequencies for a feature discovered by a model monitoring instance
    """

    def __init__(self, client, name=None, violation=None, trainingNullFreq=None, predictionNullFreq=None):
        super().__init__(client, None)
        self.name = name
        self.violation = violation
        self.training_null_freq = trainingNullFreq
        self.prediction_null_freq = predictionNullFreq

    def __repr__(self):
        return f"NullViolation(name={repr(self.name)},\n  violation={repr(self.violation)},\n  training_null_freq={repr(self.training_null_freq)},\n  prediction_null_freq={repr(self.prediction_null_freq)})"

    def to_dict(self):
        return {'name': self.name, 'violation': self.violation, 'training_null_freq': self.training_null_freq, 'prediction_null_freq': self.prediction_null_freq}
