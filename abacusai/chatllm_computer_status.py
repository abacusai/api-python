from .return_class import AbstractApiClass


class ChatllmComputerStatus(AbstractApiClass):
    """
        ChatLLM Computer Status

        Args:
            client (ApiClient): An authenticated API Client instance
            computerId (str): The ID of the computer, it can be a deployment_conversation_id or a computer_id (TODO: add separate field for deployment_conversation_id)
            vncEndpoint (str): The VNC endpoint of the computer
            computerStarted (bool): Whether the computer has started
            restartRequired (bool): Whether the computer needs to be restarted
    """

    def __init__(self, client, computerId=None, vncEndpoint=None, computerStarted=None, restartRequired=None):
        super().__init__(client, None)
        self.computer_id = computerId
        self.vnc_endpoint = vncEndpoint
        self.computer_started = computerStarted
        self.restart_required = restartRequired
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'computer_id': repr(self.computer_id), f'vnc_endpoint': repr(
            self.vnc_endpoint), f'computer_started': repr(self.computer_started), f'restart_required': repr(self.restart_required)}
        class_name = "ChatllmComputerStatus"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'computer_id': self.computer_id, 'vnc_endpoint': self.vnc_endpoint,
                'computer_started': self.computer_started, 'restart_required': self.restart_required}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
