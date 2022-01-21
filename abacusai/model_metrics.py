from .return_class import AbstractApiClass


class ModelMetrics(AbstractApiClass):
    """
        Metrics of the trained model.
    """

    def __init__(self, client, modelId=None, modelVersion=None, metrics=None, baselineMetrics=None, targetColumn=None):
        super().__init__(client, None)
        self.model_id = modelId
        self.model_version = modelVersion
        self.metrics = metrics
        self.baseline_metrics = baselineMetrics
        self.target_column = targetColumn

    def __repr__(self):
        return f"ModelMetrics(model_id={repr(self.model_id)},\n  model_version={repr(self.model_version)},\n  metrics={repr(self.metrics)},\n  baseline_metrics={repr(self.baseline_metrics)},\n  target_column={repr(self.target_column)})"

    def to_dict(self):
        return {'model_id': self.model_id, 'model_version': self.model_version, 'metrics': self.metrics, 'baseline_metrics': self.baseline_metrics, 'target_column': self.target_column}
