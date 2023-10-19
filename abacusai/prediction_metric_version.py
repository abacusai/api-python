from .return_class import AbstractApiClass


class PredictionMetricVersion(AbstractApiClass):
    """
        A Prediction Metric version for a Prediction Metric job description.

        Args:
            client (ApiClient): An authenticated API Client instance
            createdAt (str): Date and time when this prediction metric version was created.
            error (str): Relevant error if the status of this job version is FAILED.
            featureGroupVersion (str): The feature group version used as input to this prediction metric.
            predictionMetricCompletedAt (str): The time and date computations were completed for this job version.
            predictionMetricConfig (dict): Specification for the prediction metric used to run this job.
            predictionMetricId (str): The unique identifier of the prediction metric this is a version of.
            predictionMetricStartedAt (str): The time and date computations were started for this job version.
            predictionMetricVersion (str): The unique identifier of this prediction metric version.
            status (str): The current status of the computations for this job version.
    """

    def __init__(self, client, createdAt=None, error=None, featureGroupVersion=None, predictionMetricCompletedAt=None, predictionMetricConfig=None, predictionMetricId=None, predictionMetricStartedAt=None, predictionMetricVersion=None, status=None):
        super().__init__(client, predictionMetricVersion)
        self.created_at = createdAt
        self.error = error
        self.feature_group_version = featureGroupVersion
        self.prediction_metric_completed_at = predictionMetricCompletedAt
        self.prediction_metric_config = predictionMetricConfig
        self.prediction_metric_id = predictionMetricId
        self.prediction_metric_started_at = predictionMetricStartedAt
        self.prediction_metric_version = predictionMetricVersion
        self.status = status

    def __repr__(self):
        repr_dict = {f'created_at': repr(self.created_at), f'error': repr(self.error), f'feature_group_version': repr(self.feature_group_version), f'prediction_metric_completed_at': repr(self.prediction_metric_completed_at), f'prediction_metric_config': repr(
            self.prediction_metric_config), f'prediction_metric_id': repr(self.prediction_metric_id), f'prediction_metric_started_at': repr(self.prediction_metric_started_at), f'prediction_metric_version': repr(self.prediction_metric_version), f'status': repr(self.status)}
        class_name = "PredictionMetricVersion"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'created_at': self.created_at, 'error': self.error, 'feature_group_version': self.feature_group_version, 'prediction_metric_completed_at': self.prediction_metric_completed_at, 'prediction_metric_config': self.prediction_metric_config,
                'prediction_metric_id': self.prediction_metric_id, 'prediction_metric_started_at': self.prediction_metric_started_at, 'prediction_metric_version': self.prediction_metric_version, 'status': self.status}
        return {key: value for key, value in resp.items() if value is not None}

    def wait_for_prediction_metric_version(self, timeout=1200):
        """
        A waiting call until the prediction metric version is ready.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'RUNNING'}, timeout=timeout)

    def get_status(self):
        """
        Get the lifecycle status of this version.

        Returns:
            str: An enum value of the lifecycle status of this version.
        """
        return self.describe().status
