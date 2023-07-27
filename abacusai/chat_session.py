from .ai_building_task import AiBuildingTask
from .chat_message import ChatMessage
from .return_class import AbstractApiClass


class ChatSession(AbstractApiClass):
    """
        A chat session with Abacus Chat.

        Args:
            client (ApiClient): An authenticated API Client instance
            answer (str): The response from the chatbot
            chatSessionId (str): The chat session id
            projectId (str): The project id associated with the chat session
            createdAt (str): The timestamp at which the chat session was created
            chatHistory (ChatMessage): The chat history for the conversation
            nextAiBuildingTask (AiBuildingTask): The next AI building task for the chat session
    """

    def __init__(self, client, answer=None, chatSessionId=None, projectId=None, createdAt=None, chatHistory={}, nextAiBuildingTask={}):
        super().__init__(client, chatSessionId)
        self.answer = answer
        self.chat_session_id = chatSessionId
        self.project_id = projectId
        self.created_at = createdAt
        self.chat_history = client._build_class(ChatMessage, chatHistory)
        self.next_ai_building_task = client._build_class(
            AiBuildingTask, nextAiBuildingTask)

    def __repr__(self):
        return f"ChatSession(answer={repr(self.answer)},\n  chat_session_id={repr(self.chat_session_id)},\n  project_id={repr(self.project_id)},\n  created_at={repr(self.created_at)},\n  chat_history={repr(self.chat_history)},\n  next_ai_building_task={repr(self.next_ai_building_task)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'answer': self.answer, 'chat_session_id': self.chat_session_id, 'project_id': self.project_id, 'created_at': self.created_at, 'chat_history': self._get_attribute_as_dict(self.chat_history), 'next_ai_building_task': self._get_attribute_as_dict(self.next_ai_building_task)}

    def get(self):
        """
        Gets a chat session from Abacus Chat.

        Args:
            chat_session_id (str): The chat session id

        Returns:
            ChatSession: The chat session with Abacus Chat
        """
        return self.client.get_chat_session(self.chat_session_id)
