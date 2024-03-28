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
            feedbackType (str): The type of feedback provided for the message
            docInfos (list): A list of information on the documents associated with the message
            keywordArguments (dict): User message only. A dictionary of keyword arguments used to generate response.
            inputParams (dict): User message only. A dictionary of input parameters used to generate response.
            attachments (list): A list of attachments associated with the message.
            responseVersion (str): The version of the response, used to differentiate w/ legacy agent response.
            chatType (str): The type of chat llm that was run for the message.
            agentResponse (dict): Response from the agent. Only for conversation with agents.
            error (str): The error message in case of an error.
            segments (list): The segments of the message.
    """

    def __init__(self, client, role=None, text=None, timestamp=None, messageIndex=None, regenerateAttempt=None, modelVersion=None, searchResults=None, isUseful=None, feedback=None, feedbackType=None, docInfos=None, keywordArguments=None, inputParams=None, attachments=None, responseVersion=None, chatType=None, agentResponse=None, error=None, segments=None):
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
        self.feedback_type = feedbackType
        self.doc_infos = docInfos
        self.keyword_arguments = keywordArguments
        self.input_params = inputParams
        self.attachments = attachments
        self.response_version = responseVersion
        self.chat_type = chatType
        self.agent_response = agentResponse
        self.error = error
        self.segments = segments
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'role': repr(self.role), f'text': repr(self.text), f'timestamp': repr(self.timestamp), f'message_index': repr(self.message_index), f'regenerate_attempt': repr(self.regenerate_attempt), f'model_version': repr(self.model_version), f'search_results': repr(self.search_results), f'is_useful': repr(self.is_useful), f'feedback': repr(self.feedback), f'feedback_type': repr(
            self.feedback_type), f'doc_infos': repr(self.doc_infos), f'keyword_arguments': repr(self.keyword_arguments), f'input_params': repr(self.input_params), f'attachments': repr(self.attachments), f'response_version': repr(self.response_version), f'chat_type': repr(self.chat_type), f'agent_response': repr(self.agent_response), f'error': repr(self.error), f'segments': repr(self.segments)}
        class_name = "DeploymentConversationEvent"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'role': self.role, 'text': self.text, 'timestamp': self.timestamp, 'message_index': self.message_index, 'regenerate_attempt': self.regenerate_attempt, 'model_version': self.model_version, 'search_results': self.search_results, 'is_useful': self.is_useful, 'feedback': self.feedback, 'feedback_type': self.feedback_type,
                'doc_infos': self.doc_infos, 'keyword_arguments': self.keyword_arguments, 'input_params': self.input_params, 'attachments': self.attachments, 'response_version': self.response_version, 'chat_type': self.chat_type, 'agent_response': self.agent_response, 'error': self.error, 'segments': self.segments}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
