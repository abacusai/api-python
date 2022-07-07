from .return_class import AbstractApiClass


class ModelMetrics(AbstractApiClass):
    """
        Metrics of the trained model.

        Args:
            client (ApiClient): An authenticated API Client instance
            modelId (str): The Model
            modelVersion (str): The Model Version
            metrics (dict): Key/value pairs of metrics and their values
            baselineMetrics (dict): Key/value pairs of the baseline model metrics and their values
            targetColumn (str): The target column the model is predicting
            trainValTestSplit (dict): 
            infoLogs (list): 
    """

    def __init__(self, client, modelId=None, modelVersion=None, metrics=None, baselineMetrics=None, targetColumn=None, trainValTestSplit=None, infoLogs=None):
        super().__init__(client, None)
        self.model_id = modelId
        self.model_version = modelVersion
        self.metrics = metrics
        self.baseline_metrics = baselineMetrics
        self.target_column = targetColumn
        self.train_val_test_split = trainValTestSplit
        self.info_logs = infoLogs

    def __repr__(self):
        return f"ModelMetrics(model_id={repr(self.model_id)},\n  model_version={repr(self.model_version)},\n  metrics={repr(self.metrics)},\n  baseline_metrics={repr(self.baseline_metrics)},\n  target_column={repr(self.target_column)},\n  train_val_test_split={repr(self.train_val_test_split)},\n  info_logs={repr(self.info_logs)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'model_id': self.model_id, 'model_version': self.model_version, 'metrics': self.metrics, 'baseline_metrics': self.baseline_metrics, 'target_column': self.target_column, 'train_val_test_split': self.train_val_test_split, 'info_logs': self.info_logs}
