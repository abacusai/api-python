from .return_class import AbstractApiClass


class ChatMessage(AbstractApiClass):
    """
        A single chat message with Abacus Chat.

        Args:
            client (ApiClient): An authenticated API Client instance
            role (str): The role of the message sender
            text (list[dict]): A list of text segments for the message
            timestamp (str): The timestamp at which the message was sent
            isUseful (bool): Whether this message was marked as useful or not
            feedback (str): The feedback provided for the message
            docIds (list[str]): A list of IDs of the uploaded document if the message has
            hotkeyTitle (str): The title of the hotkey prompt if the message has one
            tasks (list[str]): The list of spawned tasks, if the message was broken down into smaller sub-tasks.
            keywordArguments (dict): A dict of kwargs used to generate the response.
            computePointsUsed (int): The number of compute points used for the message.
    """

    def __init__(self, client, role=None, text=None, timestamp=None, isUseful=None, feedback=None, docIds=None, hotkeyTitle=None, tasks=None, keywordArguments=None, computePointsUsed=None):
        super().__init__(client, None)
        self.role = role
        self.text = text
        self.timestamp = timestamp
        self.is_useful = isUseful
        self.feedback = feedback
        self.doc_ids = docIds
        self.hotkey_title = hotkeyTitle
        self.tasks = tasks
        self.keyword_arguments = keywordArguments
        self.compute_points_used = computePointsUsed
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'role': repr(self.role), f'text': repr(self.text), f'timestamp': repr(self.timestamp), f'is_useful': repr(self.is_useful), f'feedback': repr(self.feedback), f'doc_ids': repr(
            self.doc_ids), f'hotkey_title': repr(self.hotkey_title), f'tasks': repr(self.tasks), f'keyword_arguments': repr(self.keyword_arguments), f'compute_points_used': repr(self.compute_points_used)}
        class_name = "ChatMessage"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'role': self.role, 'text': self.text, 'timestamp': self.timestamp, 'is_useful': self.is_useful, 'feedback': self.feedback, 'doc_ids': self.doc_ids,
                'hotkey_title': self.hotkey_title, 'tasks': self.tasks, 'keyword_arguments': self.keyword_arguments, 'compute_points_used': self.compute_points_used}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
