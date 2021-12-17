from .return_class import AbstractApiClass


class CategoricalRangeViolation(AbstractApiClass):
    """
        Summary of important range mismatches for a numerical feature discovered by a model monitoring instance
    """

    def __init__(self, client, name=None, mostCommonValues=None, freqOutsideTrainingRange=None):
        super().__init__(client, None)
        self.name = name
        self.most_common_values = mostCommonValues
        self.freq_outside_training_range = freqOutsideTrainingRange

    def __repr__(self):
        return f"CategoricalRangeViolation(name={repr(self.name)},\n  most_common_values={repr(self.most_common_values)},\n  freq_outside_training_range={repr(self.freq_outside_training_range)})"

    def to_dict(self):
        return {'name': self.name, 'most_common_values': self.most_common_values, 'freq_outside_training_range': self.freq_outside_training_range}
