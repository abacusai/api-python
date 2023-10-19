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
        repr_dict = {f'type': repr(self.type), f'training_distribution': repr(self.training_distribution), f'prediction_distribution': repr(self.prediction_distribution), f'numerical_training_distribution': repr(
            self.numerical_training_distribution), f'numerical_prediction_distribution': repr(self.numerical_prediction_distribution), f'training_statistics': repr(self.training_statistics), f'prediction_statistics': repr(self.prediction_statistics)}
        class_name = "FeatureDistribution"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'type': self.type, 'training_distribution': self.training_distribution, 'prediction_distribution': self.prediction_distribution, 'numerical_training_distribution': self.numerical_training_distribution,
                'numerical_prediction_distribution': self.numerical_prediction_distribution, 'training_statistics': self.training_statistics, 'prediction_statistics': self.prediction_statistics}
        return {key: value for key, value in resp.items() if value is not None}
