from .return_class import AbstractApiClass


class HoldoutAnalysisVersion(AbstractApiClass):
    """
        A holdout analysis version object.

        Args:
            client (ApiClient): An authenticated API Client instance
            holdoutAnalysisVersion (str): The unique identifier of the holdout analysis version.
            holdoutAnalysisId (str): The unique identifier of the holdout analysis.
            createdAt (str): The timestamp at which the holdout analysis version was created.
            status (str): The status of the holdout analysis version.
            error (str): The error message if the status is FAILED.
            modelId (str): The model id associated with the holdout analysis.
            modelVersion (str): The model version associated with the holdout analysis.
            algorithm (str): The algorithm used to train the model.
            algoName (str): The name of the algorithm used to train the model.
            metrics (dict): The metrics of the holdout analysis version.
            metricInfos (dict): The metric infos of the holdout analysis version.
    """

    def __init__(self, client, holdoutAnalysisVersion=None, holdoutAnalysisId=None, createdAt=None, status=None, error=None, modelId=None, modelVersion=None, algorithm=None, algoName=None, metrics=None, metricInfos=None):
        super().__init__(client, holdoutAnalysisVersion)
        self.holdout_analysis_version = holdoutAnalysisVersion
        self.holdout_analysis_id = holdoutAnalysisId
        self.created_at = createdAt
        self.status = status
        self.error = error
        self.model_id = modelId
        self.model_version = modelVersion
        self.algorithm = algorithm
        self.algo_name = algoName
        self.metrics = metrics
        self.metric_infos = metricInfos

    def __repr__(self):
        repr_dict = {f'holdout_analysis_version': repr(self.holdout_analysis_version), f'holdout_analysis_id': repr(self.holdout_analysis_id), f'created_at': repr(self.created_at), f'status': repr(self.status), f'error': repr(
            self.error), f'model_id': repr(self.model_id), f'model_version': repr(self.model_version), f'algorithm': repr(self.algorithm), f'algo_name': repr(self.algo_name), f'metrics': repr(self.metrics), f'metric_infos': repr(self.metric_infos)}
        class_name = "HoldoutAnalysisVersion"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'holdout_analysis_version': self.holdout_analysis_version, 'holdout_analysis_id': self.holdout_analysis_id, 'created_at': self.created_at, 'status': self.status, 'error': self.error,
                'model_id': self.model_id, 'model_version': self.model_version, 'algorithm': self.algorithm, 'algo_name': self.algo_name, 'metrics': self.metrics, 'metric_infos': self.metric_infos}
        return {key: value for key, value in resp.items() if value is not None}

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            HoldoutAnalysisVersion: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self, get_metrics: bool = False):
        """
        Get a holdout analysis version.

        Args:
            get_metrics (bool): (optional) Whether to get the metrics for the holdout analysis version

        Returns:
            HoldoutAnalysisVersion: The holdout analysis version
        """
        return self.client.describe_holdout_analysis_version(self.holdout_analysis_version, get_metrics)

    def wait_for_results(self, timeout=3600):
        """
        A waiting call until holdout analysis for the version is complete

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'PROCESSING'}, timeout=timeout)

    def get_status(self):
        """
        Gets the status of the holdout analysis version.

        Returns:
            str: A string describing the status of a holdout analysis version (pending, complete, etc.).
        """
        return self.describe().status
