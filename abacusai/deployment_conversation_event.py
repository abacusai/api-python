from .return_class import AbstractApiClass


class DeploymentConversationEvent(AbstractApiClass):
    """
        A single deployment conversation message.

        Args:
            client (ApiClient): An authenticated API Client instance
            role (str): The role of the message sender
            text (str): The text of the message
            timestamp (str): The timestamp at which the message was sent
            modelVersion (str): The model instance id associated with the message.
            searchResults (dict): The search results for the message.
            isUseful (bool): Whether this message was marked as useful or not
            feedback (str): The feedback provided for the message
    """

    def __init__(self, client, role=None, text=None, timestamp=None, modelVersion=None, searchResults=None, isUseful=None, feedback=None):
        super().__init__(client, None)
        self.role = role
        self.text = text
        self.timestamp = timestamp
        self.model_version = modelVersion
        self.search_results = searchResults
        self.is_useful = isUseful
        self.feedback = feedback

    def __repr__(self):
        return f"DeploymentConversationEvent(role={repr(self.role)},\n  text={repr(self.text)},\n  timestamp={repr(self.timestamp)},\n  model_version={repr(self.model_version)},\n  search_results={repr(self.search_results)},\n  is_useful={repr(self.is_useful)},\n  feedback={repr(self.feedback)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'role': self.role, 'text': self.text, 'timestamp': self.timestamp, 'model_version': self.model_version, 'search_results': self.search_results, 'is_useful': self.is_useful, 'feedback': self.feedback}
