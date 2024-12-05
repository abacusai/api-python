from .return_class import AbstractApiClass


class ChatllmComputer(AbstractApiClass):
    """
        ChatLLMComputer

        Args:
            client (ApiClient): An authenticated API Client instance
            computerId (int): The computer id.
            token (str): The token.
            vncEndpoint (str): The VNC endpoint.
    """

    def __init__(self, client, computerId=None, token=None, vncEndpoint=None):
        super().__init__(client, None)
        self.computer_id = computerId
        self.token = token
        self.vnc_endpoint = vncEndpoint
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'computer_id': repr(self.computer_id), f'token': repr(
            self.token), f'vnc_endpoint': repr(self.vnc_endpoint)}
        class_name = "ChatllmComputer"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'computer_id': self.computer_id,
                'token': self.token, 'vnc_endpoint': self.vnc_endpoint}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
