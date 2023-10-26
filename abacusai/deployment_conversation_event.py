from .return_class import AbstractApiClass


class DeploymentConversationEvent(AbstractApiClass):
    """
        A single deployment conversation message.

        Args:
            client (ApiClient): An authenticated API Client instance
            role (str): The role of the message sender
            text (str): The text of the message
            timestamp (str): The timestamp at which the message was sent
            messageIndex (int): The index of the message in the conversation
            regenerateAttempt (int): The sequence number of regeneration. Not regenerated if 0.
            modelVersion (str): The model instance id associated with the message.
            searchResults (dict): The search results for the message.
            isUseful (bool): Whether this message was marked as useful or not
            feedback (str): The feedback provided for the message
            docId (str): The document id associated with the message
    """

    def __init__(self, client, role=None, text=None, timestamp=None, messageIndex=None, regenerateAttempt=None, modelVersion=None, searchResults=None, isUseful=None, feedback=None, docId=None):
        super().__init__(client, None)
        self.role = role
        self.text = text
        self.timestamp = timestamp
        self.message_index = messageIndex
        self.regenerate_attempt = regenerateAttempt
        self.model_version = modelVersion
        self.search_results = searchResults
        self.is_useful = isUseful
        self.feedback = feedback
        self.doc_id = docId

    def __repr__(self):
        repr_dict = {f'role': repr(self.role), f'text': repr(self.text), f'timestamp': repr(self.timestamp), f'message_index': repr(self.message_index), f'regenerate_attempt': repr(
            self.regenerate_attempt), f'model_version': repr(self.model_version), f'search_results': repr(self.search_results), f'is_useful': repr(self.is_useful), f'feedback': repr(self.feedback), f'doc_id': repr(self.doc_id)}
        class_name = "DeploymentConversationEvent"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'role': self.role, 'text': self.text, 'timestamp': self.timestamp, 'message_index': self.message_index, 'regenerate_attempt': self.regenerate_attempt,
                'model_version': self.model_version, 'search_results': self.search_results, 'is_useful': self.is_useful, 'feedback': self.feedback, 'doc_id': self.doc_id}
        return {key: value for key, value in resp.items() if value is not None}
