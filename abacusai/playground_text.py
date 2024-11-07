from .return_class import AbstractApiClass


class PlaygroundText(AbstractApiClass):
    """
        The text content inside of a playground segment.

        Args:
            client (ApiClient): An authenticated API Client instance
            playgroundText (str): The text of the playground segment.
            renderingCode (str): The rendering code of the playground segment.
    """

    def __init__(self, client, playgroundText=None, renderingCode=None):
        super().__init__(client, None)
        self.playground_text = playgroundText
        self.rendering_code = renderingCode
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'playground_text': repr(
            self.playground_text), f'rendering_code': repr(self.rendering_code)}
        class_name = "PlaygroundText"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'playground_text': self.playground_text,
                'rendering_code': self.rendering_code}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
