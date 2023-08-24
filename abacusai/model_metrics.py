from .return_class import AbstractApiClass


class ModelMetrics(AbstractApiClass):
    """
        Metrics of the trained model.

        Args:
            client (ApiClient): An authenticated API Client instance
            algoMetrics (dict): Dictionary mapping algorithm ID to algorithm name and algorithm metrics dictionary
            selectedAlgorithm (str): The algorithm ID of the selected (default) algorithm that will be used in deployments of this Model Version
            modelId (str): The Model ID
            modelVersion (str): The Model Version
            metricNames (dict): Maps shorthand names of the metrics to their verbose names
            targetColumn (str): The target feature that the model was trained to predict
            trainValTestSplit (dict): Info on train, val and test split
            trainingCompletedAt (datetime): Timestamp when training was completed
            baselineMetrics (dict): Key/value pairs of the baseline model metrics and their values
    """

    def __init__(self, client, algoMetrics=None, selectedAlgorithm=None, modelId=None, modelVersion=None, metricNames=None, targetColumn=None, trainValTestSplit=None, trainingCompletedAt=None, baselineMetrics=None):
        super().__init__(client, None)
        self.algo_metrics = algoMetrics
        self.selected_algorithm = selectedAlgorithm
        self.model_id = modelId
        self.model_version = modelVersion
        self.metric_names = metricNames
        self.target_column = targetColumn
        self.train_val_test_split = trainValTestSplit
        self.training_completed_at = trainingCompletedAt
        self.baseline_metrics = baselineMetrics

    def __repr__(self):
        return f"ModelMetrics(algo_metrics={repr(self.algo_metrics)},\n  selected_algorithm={repr(self.selected_algorithm)},\n  model_id={repr(self.model_id)},\n  model_version={repr(self.model_version)},\n  metric_names={repr(self.metric_names)},\n  target_column={repr(self.target_column)},\n  train_val_test_split={repr(self.train_val_test_split)},\n  training_completed_at={repr(self.training_completed_at)},\n  baseline_metrics={repr(self.baseline_metrics)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'algo_metrics': self.algo_metrics, 'selected_algorithm': self.selected_algorithm, 'model_id': self.model_id, 'model_version': self.model_version, 'metric_names': self.metric_names, 'target_column': self.target_column, 'train_val_test_split': self.train_val_test_split, 'training_completed_at': self.training_completed_at, 'baseline_metrics': self.baseline_metrics}
