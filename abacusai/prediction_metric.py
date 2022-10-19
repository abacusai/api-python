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
            predictionMetricConfig (json): Specification for the prediction metric to run in this job.
            predictionMetricId (str): The unique identifier of the prediction metric.
            modelMonitorId (str): unique string identifier for model monitor that created prediction metric
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

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            PredictionMetric: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self, should_include_latest_version_description: bool = True):
        """
        Describe a Prediction Metric.

        Args:
            should_include_latest_version_description (bool): include the description of the latest prediction metric version

        Returns:
            PredictionMetric: The prediction metric object.
        """
        return self.client.describe_prediction_metric(self.prediction_metric_id, should_include_latest_version_description)

    def delete(self):
        """
        Removes an existing PredictionMetric.

        Args:
            prediction_metric_id (str): The unique ID associated with the prediction metric.
        """
        return self.client.delete_prediction_metric(self.prediction_metric_id)

    def run(self):
        """
        Creates a new prediction metrics job run for the given prediction metric job description, and starts that job.

        Configures and starts the computations running to compute the prediciton metric.


        Args:
            prediction_metric_id (str): The prediction metric job description to apply for configuring a prediction metric job.

        Returns:
            PredictionMetricVersion: A prediction metric version. For more information, please refer to the details on the object (below).
        """
        return self.client.run_prediction_metric(self.prediction_metric_id)

    def list_versions(self, limit: int = 100, start_after_id: str = None):
        """
        List the prediction metric versions for a prediction metric.

        Args:
            limit (int): The the number of prediction metric instances to be retrieved.
            start_after_id (str): An offset parameter to exclude all prediction metric versions till the specified prediction metric ID.

        Returns:
            PredictionMetricVersion: The prediction metric instances for this prediction metric.
        """
        return self.client.list_prediction_metric_versions(self.prediction_metric_id, limit, start_after_id)
