from .return_class import AbstractApiClass


class FeatureDistribution(AbstractApiClass):
    """
        For a single feature, how it has changed in the training data versus some specified window

        Args:
            client (ApiClient): An authenticated API Client instance
            type (str): Data type of values in each distribution, typically 'categorical' or 'numerical'.
            trainingDistribution (dict): A dict describing the range of values in the training distribution.
            predictionDistribution (dict): A dict describing the range of values in the specified window.
            numericalTrainingDistribution (dict): A dict describing the summary statistics of the numerical training distribution.
            numericalPredictionDistribution (dict): A dict describing the summary statistics of the numerical prediction distribution.
            trainingStatistics (dict): A dict describing summary statistics of values in the training distribution.
            predictionStatistics (dict): A dict describing summary statistics of values in the specified window.
    """

    def __init__(self, client, type=None, trainingDistribution=None, predictionDistribution=None, numericalTrainingDistribution=None, numericalPredictionDistribution=None, trainingStatistics=None, predictionStatistics=None):
        super().__init__(client, None)
        self.type = type
        self.training_distribution = trainingDistribution
        self.prediction_distribution = predictionDistribution
        self.numerical_training_distribution = numericalTrainingDistribution
        self.numerical_prediction_distribution = numericalPredictionDistribution
        self.training_statistics = trainingStatistics
        self.prediction_statistics = predictionStatistics

    def __repr__(self):
        return f"FeatureDistribution(type={repr(self.type)},\n  training_distribution={repr(self.training_distribution)},\n  prediction_distribution={repr(self.prediction_distribution)},\n  numerical_training_distribution={repr(self.numerical_training_distribution)},\n  numerical_prediction_distribution={repr(self.numerical_prediction_distribution)},\n  training_statistics={repr(self.training_statistics)},\n  prediction_statistics={repr(self.prediction_statistics)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'type': self.type, 'training_distribution': self.training_distribution, 'prediction_distribution': self.prediction_distribution, 'numerical_training_distribution': self.numerical_training_distribution, 'numerical_prediction_distribution': self.numerical_prediction_distribution, 'training_statistics': self.training_statistics, 'prediction_statistics': self.prediction_statistics}
