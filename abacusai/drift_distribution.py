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
        return f"DriftDistribution(train_column={repr(self.train_column)},\n  predicted_column={repr(self.predicted_column)},\n  metrics={repr(self.metrics)},\n  distribution={repr(self.distribution)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'train_column': self.train_column, 'predicted_column': self.predicted_column, 'metrics': self.metrics, 'distribution': self._get_attribute_as_dict(self.distribution)}
