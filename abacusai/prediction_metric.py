from .prediction_metric_version import PredictionMetricVersion
from .refresh_schedule import RefreshSchedule
from .return_class import AbstractApiClass


class PredictionMetric(AbstractApiClass):
    """
        A Prediction Metric job description.

        Args:
            client (ApiClient): An authenticated API Client instance
            createdAt (str): Date and time when this prediction metric was created.
            featureGroupId (str): The feature group used as input to this prediction metric.
            predictionMetricConfig (dict): Specification for the prediction metric to run in this job.
            predictionMetricId (str): The unique identifier of the prediction metric.
            modelMonitorId (str): The unique string identifier for model monitor that created this prediction metric
            projectId (str): The project this prediction metric belongs to.
            latestPredictionMetricVersionDescription (PredictionMetricVersion): Description of the latest prediction metric version (if any).
            refreshSchedules (RefreshSchedule): List of schedules that determines when the next version of the dataset will be created.
    """

    def __init__(self, client, createdAt=None, featureGroupId=None, predictionMetricConfig=None, predictionMetricId=None, modelMonitorId=None, projectId=None, refreshSchedules={}, latestPredictionMetricVersionDescription={}):
        super().__init__(client, predictionMetricId)
        self.created_at = createdAt
        self.feature_group_id = featureGroupId
        self.prediction_metric_config = predictionMetricConfig
        self.prediction_metric_id = predictionMetricId
        self.model_monitor_id = modelMonitorId
        self.project_id = projectId
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)
        self.latest_prediction_metric_version_description = client._build_class(
            PredictionMetricVersion, latestPredictionMetricVersionDescription)

    def __repr__(self):
        return f"PredictionMetric(created_at={repr(self.created_at)},\n  feature_group_id={repr(self.feature_group_id)},\n  prediction_metric_config={repr(self.prediction_metric_config)},\n  prediction_metric_id={repr(self.prediction_metric_id)},\n  model_monitor_id={repr(self.model_monitor_id)},\n  project_id={repr(self.project_id)},\n  refresh_schedules={repr(self.refresh_schedules)},\n  latest_prediction_metric_version_description={repr(self.latest_prediction_metric_version_description)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'created_at': self.created_at, 'feature_group_id': self.feature_group_id, 'prediction_metric_config': self.prediction_metric_config, 'prediction_metric_id': self.prediction_metric_id, 'model_monitor_id': self.model_monitor_id, 'project_id': self.project_id, 'refresh_schedules': self._get_attribute_as_dict(self.refresh_schedules), 'latest_prediction_metric_version_description': self._get_attribute_as_dict(self.latest_prediction_metric_version_description)}
