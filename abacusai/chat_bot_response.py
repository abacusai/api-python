from .return_class import AbstractApiClass


class ChatBotResponse(AbstractApiClass):
    """
        A response from the chat bot

        Args:
            client (ApiClient): An authenticated API Client instance
            answer (str): The response from the chat bot
            chatHistory (list): The chat history as a list of dicts with is_user and text entries
    """

    def __init__(self, client, answer=None, chatHistory=None):
        super().__init__(client, None)
        self.answer = answer
        self.chat_history = chatHistory

    def __repr__(self):
        return f"ChatBotResponse(answer={repr(self.answer)},\n  chat_history={repr(self.chat_history)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'answer': self.answer, 'chat_history': self.chat_history}
