from .feature_distribution import FeatureDistribution
from .return_class import AbstractApiClass


class DriftDistribution(AbstractApiClass):
    """
        How actuals or predicted values have changed in the training data versus predicted data

        Args:
            client (ApiClient): An authenticated API Client instance
            trainColumn (str): The feature name in the train table.
            predictedColumn (str): The feature name in the prediction table.
            metrics (dict): Drift measures.
            distribution (FeatureDistribution): A FeatureDistribution, how the training data compares to the predicted data.
    """

    def __init__(self, client, trainColumn=None, predictedColumn=None, metrics=None, distribution={}):
        super().__init__(client, None)
        self.train_column = trainColumn
        self.predicted_column = predictedColumn
        self.metrics = metrics
        self.distribution = client._build_class(
            FeatureDistribution, distribution)

    def __repr__(self):
        repr_dict = {f'train_column': repr(self.train_column), f'predicted_column': repr(
            self.predicted_column), f'metrics': repr(self.metrics), f'distribution': repr(self.distribution)}
        class_name = "DriftDistribution"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'train_column': self.train_column, 'predicted_column': self.predicted_column,
                'metrics': self.metrics, 'distribution': self._get_attribute_as_dict(self.distribution)}
        return {key: value for key, value in resp.items() if value is not None}
