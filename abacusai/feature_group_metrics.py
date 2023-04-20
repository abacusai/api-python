from .return_class import AbstractApiClass


class FeatureGroupMetrics(AbstractApiClass):
    """
        Metrics for a feature group

        Args:
            client (ApiClient): An authenticated API Client instance
            metrics (list[dict]): A list of dicts with metrics for each columns in the feature group
            schema (list[dict]): A list of dicts with the schema for each metric
            numRows (int): The number of rows in the feature group
            numCols (int): The number of columns in the feature group
    """

    def __init__(self, client, metrics=None, schema=None, numRows=None, numCols=None):
        super().__init__(client, None)
        self.metrics = metrics
        self.schema = schema
        self.num_rows = numRows
        self.num_cols = numCols

    def __repr__(self):
        return f"FeatureGroupMetrics(metrics={repr(self.metrics)},\n  schema={repr(self.schema)},\n  num_rows={repr(self.num_rows)},\n  num_cols={repr(self.num_cols)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'metrics': self.metrics, 'schema': self.schema, 'num_rows': self.num_rows, 'num_cols': self.num_cols}
