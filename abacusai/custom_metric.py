from .custom_metric_version import CustomMetricVersion
from .return_class import AbstractApiClass


class CustomMetric(AbstractApiClass):
    """
        Custom metric.

        Args:
            client (ApiClient): An authenticated API Client instance
            customMetricId (str): Unique string identifier of the custom metric.
            name (str): Name assigned to the custom metric.
            createdAt (str): Date and time when the custom metric was created (ISO 8601 format).
            problemType (str): Problem type that this custom metric is applicable to (e.g. regression).
            notebookId (str): Unique string identifier of the notebook used to create/edit the custom metric.
            latestCustomMetricVersion (CustomMetricVersion): Latest version of the custom metric.
    """

    def __init__(self, client, customMetricId=None, name=None, createdAt=None, problemType=None, notebookId=None, latestCustomMetricVersion={}):
        super().__init__(client, customMetricId)
        self.custom_metric_id = customMetricId
        self.name = name
        self.created_at = createdAt
        self.problem_type = problemType
        self.notebook_id = notebookId
        self.latest_custom_metric_version = client._build_class(
            CustomMetricVersion, latestCustomMetricVersion)

    def __repr__(self):
        return f"CustomMetric(custom_metric_id={repr(self.custom_metric_id)},\n  name={repr(self.name)},\n  created_at={repr(self.created_at)},\n  problem_type={repr(self.problem_type)},\n  notebook_id={repr(self.notebook_id)},\n  latest_custom_metric_version={repr(self.latest_custom_metric_version)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'custom_metric_id': self.custom_metric_id, 'name': self.name, 'created_at': self.created_at, 'problem_type': self.problem_type, 'notebook_id': self.notebook_id, 'latest_custom_metric_version': self._get_attribute_as_dict(self.latest_custom_metric_version)}
