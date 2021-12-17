from .return_class import AbstractApiClass


class PredictionMetric(AbstractApiClass):
    """
        A Prediction Metric job description.
    """

    def __init__(self, client, createdAt=None, featureGroupId=None, predictionMetricConfig=None, predictionMetricId=None, projectId=None):
        super().__init__(client, predictionMetricId)
        self.created_at = createdAt
        self.feature_group_id = featureGroupId
        self.prediction_metric_config = predictionMetricConfig
        self.prediction_metric_id = predictionMetricId
        self.project_id = projectId

    def __repr__(self):
        return f"PredictionMetric(created_at={repr(self.created_at)},\n  feature_group_id={repr(self.feature_group_id)},\n  prediction_metric_config={repr(self.prediction_metric_config)},\n  prediction_metric_id={repr(self.prediction_metric_id)},\n  project_id={repr(self.project_id)})"

    def to_dict(self):
        return {'created_at': self.created_at, 'feature_group_id': self.feature_group_id, 'prediction_metric_config': self.prediction_metric_config, 'prediction_metric_id': self.prediction_metric_id, 'project_id': self.project_id}

    def refresh(self):
        """Calls describe and refreshes the current object's fields"""
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """Describe a Prediction Metric."""
        return self.client.describe_prediction_metric(self.prediction_metric_id)

    def delete(self):
        """Removes an existing PredictionMetric."""
        return self.client.delete_prediction_metric(self.prediction_metric_id)

    def run(self):
        """Creates a new prediction metrics job run for the given prediction metric job description, and starts that job."""
        return self.client.run_prediction_metric(self.prediction_metric_id)

    def list_versions(self, limit=100, start_after_id=None):
        """List the prediction metric versions for a prediction metric."""
        return self.client.list_prediction_metric_versions(self.prediction_metric_id, limit, start_after_id)
