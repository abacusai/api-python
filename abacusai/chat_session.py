from .ai_building_task import AiBuildingTask
from .chat_message import ChatMessage
from .return_class import AbstractApiClass


class ChatSession(AbstractApiClass):
    """
        A chat session with Abacus Data Science Co-pilot.

        Args:
            client (ApiClient): An authenticated API Client instance
            answer (str): The response from the chatbot
            chatSessionId (str): The chat session id
            projectId (str): The project id associated with the chat session
            name (str): The name of the chat session
            createdAt (str): The timestamp at which the chat session was created
            status (str): The status of the chat sessions
            aiBuildingInProgress (bool): Whether the AI building is in progress or not
            notification (str): A warn/info message about the chat session. For example, a suggestion to create a new session if the current one is too old
            whiteboard (str): A set of whiteboard notes associated with the chat session
            chatHistory (ChatMessage): The chat history for the conversation
            nextAiBuildingTask (AiBuildingTask): The next AI building task for the chat session
    """

    def __init__(self, client, answer=None, chatSessionId=None, projectId=None, name=None, createdAt=None, status=None, aiBuildingInProgress=None, notification=None, whiteboard=None, chatHistory={}, nextAiBuildingTask={}):
        super().__init__(client, chatSessionId)
        self.answer = answer
        self.chat_session_id = chatSessionId
        self.project_id = projectId
        self.name = name
        self.created_at = createdAt
        self.status = status
        self.ai_building_in_progress = aiBuildingInProgress
        self.notification = notification
        self.whiteboard = whiteboard
        self.chat_history = client._build_class(ChatMessage, chatHistory)
        self.next_ai_building_task = client._build_class(
            AiBuildingTask, nextAiBuildingTask)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'answer': repr(self.answer), f'chat_session_id': repr(self.chat_session_id), f'project_id': repr(self.project_id), f'name': repr(self.name), f'created_at': repr(self.created_at), f'status': repr(self.status), f'ai_building_in_progress': repr(
            self.ai_building_in_progress), f'notification': repr(self.notification), f'whiteboard': repr(self.whiteboard), f'chat_history': repr(self.chat_history), f'next_ai_building_task': repr(self.next_ai_building_task)}
        class_name = "ChatSession"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'answer': self.answer, 'chat_session_id': self.chat_session_id, 'project_id': self.project_id, 'name': self.name, 'created_at': self.created_at, 'status': self.status, 'ai_building_in_progress': self.ai_building_in_progress,
                'notification': self.notification, 'whiteboard': self.whiteboard, 'chat_history': self._get_attribute_as_dict(self.chat_history), 'next_ai_building_task': self._get_attribute_as_dict(self.next_ai_building_task)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def get(self):
        """
        Gets a chat session from Data Science Co-pilot.

        Args:
            chat_session_id (str): Unique ID of the chat session.

        Returns:
            ChatSession: The chat session with Data Science Co-pilot
        """
        return self.client.get_chat_session(self.chat_session_id)

    def delete_chat_message(self, message_index: int):
        """
        Deletes a message in a chat session and its associated response.

        Args:
            message_index (int): The index of the chat message within the UI.
        """
        return self.client.delete_chat_message(self.chat_session_id, message_index)

    def export(self):
        """
        Exports a chat session to an HTML file

        Args:
            chat_session_id (str): Unique ID of the chat session.
        """
        return self.client.export_chat_session(self.chat_session_id)

    def rename(self, name: str):
        """
        Renames a chat session with Data Science Co-pilot.

        Args:
            name (str): The new name of the chat session.
        """
        return self.client.rename_chat_session(self.chat_session_id, name)
