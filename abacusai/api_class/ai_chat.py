import dataclasses

from .abstract import ApiClass


@dataclasses.dataclass
class HotkeyPrompt(ApiClass):
    """
    A config class for a Data Science Co-Pilot Hotkey

    Args:
        prompt (str): The prompt to send to Data Science Co-Pilot
        title (str): A short, descriptive title for the prompt. If not provided, one will be automatically generated.
    """
    prompt: str
    title: str = dataclasses.field(default=None)
    disable_problem_type_context: bool = dataclasses.field(default=True)
    ignore_history: bool = dataclasses.field(default=None)


@dataclasses.dataclass
class AgentConversationMessage(ApiClass):
    """
    Message format for agent conversation

    Args:
        is_user (bool): Whether the message is from the user.
        text (str): The message's text.
        document_content (str): Document text in case of any document present.
    """
    is_user: bool = dataclasses.field(default=None)
    text: str = dataclasses.field(default=None)
    document_content: str = dataclasses.field(default=None)

    def to_dict(self):
        return {
            'is_user': self.is_user,
            'text': self.text,
            'document_content': self.document_content
        }
