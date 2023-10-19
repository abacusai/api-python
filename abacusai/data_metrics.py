from .return_class import AbstractApiClass


class DataMetrics(AbstractApiClass):
    """
        Processed Metrics and Schema for a dataset version or feature group version

        Args:
            client (ApiClient): An authenticated API Client instance
            metrics (list[dict]): A list of dicts with metrics for each columns
            schema (list[dict]): A list of dicts with the schema for each metric
            numRows (int): The number of rows
            numCols (int): The number of columns
    """

    def __init__(self, client, metrics=None, schema=None, numRows=None, numCols=None):
        super().__init__(client, None)
        self.metrics = metrics
        self.schema = schema
        self.num_rows = numRows
        self.num_cols = numCols

    def __repr__(self):
        repr_dict = {f'metrics': repr(self.metrics), f'schema': repr(
            self.schema), f'num_rows': repr(self.num_rows), f'num_cols': repr(self.num_cols)}
        class_name = "DataMetrics"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'metrics': self.metrics, 'schema': self.schema,
                'num_rows': self.num_rows, 'num_cols': self.num_cols}
        return {key: value for key, value in resp.items() if value is not None}
