from .chat_message import ChatMessage
from .return_class import AbstractApiClass


class ChatSession(AbstractApiClass):
    """
        A chat session with Abacus Chat.

        Args:
            client (ApiClient): An authenticated API Client instance
            answer (str): The response from the chatbot
            availableIndices (list[dict]): A list of indices that the chatbot has access to
            chatSessionId (str): The chat session id
            projectId (str): The project id associated with the chat session
            chatHistory (ChatMessage): The chat history for the conversation
    """

    def __init__(self, client, answer=None, availableIndices=None, chatSessionId=None, projectId=None, chatHistory={}):
        super().__init__(client, chatSessionId)
        self.answer = answer
        self.available_indices = availableIndices
        self.chat_session_id = chatSessionId
        self.project_id = projectId
        self.chat_history = client._build_class(ChatMessage, chatHistory)

    def __repr__(self):
        return f"ChatSession(answer={repr(self.answer)},\n  available_indices={repr(self.available_indices)},\n  chat_session_id={repr(self.chat_session_id)},\n  project_id={repr(self.project_id)},\n  chat_history={repr(self.chat_history)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'answer': self.answer, 'available_indices': self.available_indices, 'chat_session_id': self.chat_session_id, 'project_id': self.project_id, 'chat_history': self._get_attribute_as_dict(self.chat_history)}

    def get(self):
        """
        Gets a chat session from Abacus Chat.

        Args:
            chat_session_id (str): The chat session id

        Returns:
            ChatSession: The chat session with Abacus Chat
        """
        return self.client.get_chat_session(self.chat_session_id)

    def send_chat_message(self, message: str):
        """
        Updates chat history with the response from a user message

        Args:
            message (str): Message you want to send to Abacus Chat

        Returns:
            ChatSession: The chat session with Abacus Chat
        """
        return self.client.send_chat_message(self.chat_session_id, message)
