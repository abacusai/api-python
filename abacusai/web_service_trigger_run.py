from .return_class import AbstractApiClass


class WebServiceTriggerRun(AbstractApiClass):
    """
        A web service trigger run

        Args:
            client (ApiClient): An authenticated API Client instance
            endpoint (str): The endpoint of the web service trigger run.
            createdAt (str): The creation time of the web service trigger run.
            payload (dict): The payload of the web service trigger run.
            headers (dict): The headers of the web service trigger run.
            method (str): The method of the web service trigger run.
            responseStatus (int): The HTTP response status code.
            responseBody (str): The HTTP response body.
            error (str): Error message if the request failed.
            lifecycle (str): The lifecycle status of the run.
    """

    def __init__(self, client, endpoint=None, createdAt=None, payload=None, headers=None, method=None, responseStatus=None, responseBody=None, error=None, lifecycle=None):
        super().__init__(client, None)
        self.endpoint = endpoint
        self.created_at = createdAt
        self.payload = payload
        self.headers = headers
        self.method = method
        self.response_status = responseStatus
        self.response_body = responseBody
        self.error = error
        self.lifecycle = lifecycle
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'endpoint': repr(self.endpoint), f'created_at': repr(self.created_at), f'payload': repr(self.payload), f'headers': repr(self.headers), f'method': repr(
            self.method), f'response_status': repr(self.response_status), f'response_body': repr(self.response_body), f'error': repr(self.error), f'lifecycle': repr(self.lifecycle)}
        class_name = "WebServiceTriggerRun"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'endpoint': self.endpoint, 'created_at': self.created_at, 'payload': self.payload, 'headers': self.headers, 'method': self.method,
                'response_status': self.response_status, 'response_body': self.response_body, 'error': self.error, 'lifecycle': self.lifecycle}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
