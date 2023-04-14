from .return_class import AbstractApiClass


class FeatureGroupMetrics(AbstractApiClass):
    """
        Metrics for a feature group

        Args:
            client (ApiClient): An authenticated API Client instance
            metrics (list[dict]): A list of dicts with metrics for each columns in the feature group
            schema (list[dict]): A list of dicts with the schema for each metric
    """

    def __init__(self, client, metrics=None, schema=None):
        super().__init__(client, None)
        self.metrics = metrics
        self.schema = schema

    def __repr__(self):
        return f"FeatureGroupMetrics(metrics={repr(self.metrics)},\n  schema={repr(self.schema)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'metrics': self.metrics, 'schema': self.schema}
