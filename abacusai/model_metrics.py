from .return_class import AbstractApiClass


class ModelMetrics(AbstractApiClass):
    """
        Metrics of the trained model.

        Args:
            client (ApiClient): An authenticated API Client instance
            algoMetrics (dict): Dictionary mapping algorithm ID to algorithm name and algorithm metrics dictionary
            selectedAlgorithm (str): The algorithm ID of the selected (default) algorithm that will be used in deployments of this Model Version
            selectedAlgorithmName (str): The algorithm name of the selected (default) algorithm that will be used in deployments of this Model Version
            modelId (str): The Model ID
            modelVersion (str): The Model Version
            metricNames (dict): Maps shorthand names of the metrics to their verbose names
            targetColumn (str): The target feature that the model was trained to predict
            trainValTestSplit (dict): Info on train, val and test split
            trainingCompletedAt (datetime): Timestamp when training was completed
    """

    def __init__(self, client, algoMetrics=None, selectedAlgorithm=None, selectedAlgorithmName=None, modelId=None, modelVersion=None, metricNames=None, targetColumn=None, trainValTestSplit=None, trainingCompletedAt=None):
        super().__init__(client, None)
        self.algo_metrics = algoMetrics
        self.selected_algorithm = selectedAlgorithm
        self.selected_algorithm_name = selectedAlgorithmName
        self.model_id = modelId
        self.model_version = modelVersion
        self.metric_names = metricNames
        self.target_column = targetColumn
        self.train_val_test_split = trainValTestSplit
        self.training_completed_at = trainingCompletedAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'algo_metrics': repr(self.algo_metrics), f'selected_algorithm': repr(self.selected_algorithm), f'selected_algorithm_name': repr(self.selected_algorithm_name), f'model_id': repr(self.model_id), f'model_version': repr(
            self.model_version), f'metric_names': repr(self.metric_names), f'target_column': repr(self.target_column), f'train_val_test_split': repr(self.train_val_test_split), f'training_completed_at': repr(self.training_completed_at)}
        class_name = "ModelMetrics"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'algo_metrics': self.algo_metrics, 'selected_algorithm': self.selected_algorithm, 'selected_algorithm_name': self.selected_algorithm_name, 'model_id': self.model_id, 'model_version': self.model_version,
                'metric_names': self.metric_names, 'target_column': self.target_column, 'train_val_test_split': self.train_val_test_split, 'training_completed_at': self.training_completed_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
