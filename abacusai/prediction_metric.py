from .prediction_metric_version import PredictionMetricVersion
from .return_class import AbstractApiClass


class PredictionMetric(AbstractApiClass):
    """
        A Prediction Metric job description.

        Args:
            client (ApiClient): An authenticated API Client instance
            createdAt (str): Date and time when this prediction metric was created.
            featureGroupId (str): The feature group used as input to this prediction metric.
            predictionMetricConfig (json): Specification for the prediction metric to run in this job.
            predictionMetricId (str): The unique identifier of the prediction metric.
            projectId (str): The project this prediction metric belongs to.
            latestPredictionMetricVersionDescription (PredictionMetricVersion): Description of the latest prediction metric version (if any).
    """

    def __init__(self, client, createdAt=None, featureGroupId=None, predictionMetricConfig=None, predictionMetricId=None, projectId=None, latestPredictionMetricVersionDescription={}):
        super().__init__(client, predictionMetricId)
        self.created_at = createdAt
        self.feature_group_id = featureGroupId
        self.prediction_metric_config = predictionMetricConfig
        self.prediction_metric_id = predictionMetricId
        self.project_id = projectId
        self.latest_prediction_metric_version_description = client._build_class(
            PredictionMetricVersion, latestPredictionMetricVersionDescription)

    def __repr__(self):
        return f"PredictionMetric(created_at={repr(self.created_at)},\n  feature_group_id={repr(self.feature_group_id)},\n  prediction_metric_config={repr(self.prediction_metric_config)},\n  prediction_metric_id={repr(self.prediction_metric_id)},\n  project_id={repr(self.project_id)},\n  latest_prediction_metric_version_description={repr(self.latest_prediction_metric_version_description)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'created_at': self.created_at, 'feature_group_id': self.feature_group_id, 'prediction_metric_config': self.prediction_metric_config, 'prediction_metric_id': self.prediction_metric_id, 'project_id': self.project_id, 'latest_prediction_metric_version_description': self._get_attribute_as_dict(self.latest_prediction_metric_version_description)}