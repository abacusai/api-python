from .return_class import AbstractApiClass


class CategoricalRangeViolation(AbstractApiClass):
    """
        Summary of important range mismatches for a numerical feature discovered by a model monitoring instance

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): Name of feature.
            mostCommonValues (list[str]): List of most common feature names in the prediction distribution not present in the training distribution.
            freqOutsideTrainingRange (float): Frequency of prediction rows outside training distribution for the specified feature.
    """

    def __init__(self, client, name=None, mostCommonValues=None, freqOutsideTrainingRange=None):
        super().__init__(client, None)
        self.name = name
        self.most_common_values = mostCommonValues
        self.freq_outside_training_range = freqOutsideTrainingRange

    def __repr__(self):
        return f"CategoricalRangeViolation(name={repr(self.name)},\n  most_common_values={repr(self.most_common_values)},\n  freq_outside_training_range={repr(self.freq_outside_training_range)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'most_common_values': self.most_common_values, 'freq_outside_training_range': self.freq_outside_training_range}
