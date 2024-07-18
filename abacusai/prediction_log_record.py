from .return_class import AbstractApiClass


class PredictionLogRecord(AbstractApiClass):
    """
        A Record for a prediction request log.

        Args:
            client (ApiClient): An authenticated API Client instance
            requestId (str): The unique identifier of the prediction request.
            query (dict): The query used to make the prediction.
            queryTimeMs (int): The time taken to make the prediction.
            timestampMs (str): The timestamp of the prediction request.
            response (dict): The prediction response.
    """

    def __init__(self, client, requestId=None, query=None, queryTimeMs=None, timestampMs=None, response=None):
        super().__init__(client, None)
        self.request_id = requestId
        self.query = query
        self.query_time_ms = queryTimeMs
        self.timestamp_ms = timestampMs
        self.response = response
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'request_id': repr(self.request_id), f'query': repr(self.query), f'query_time_ms': repr(
            self.query_time_ms), f'timestamp_ms': repr(self.timestamp_ms), f'response': repr(self.response)}
        class_name = "PredictionLogRecord"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'request_id': self.request_id, 'query': self.query, 'query_time_ms': self.query_time_ms,
                'timestamp_ms': self.timestamp_ms, 'response': self.response}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
