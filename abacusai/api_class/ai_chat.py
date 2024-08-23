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
