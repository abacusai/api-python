from .return_class import AbstractApiClass


class HumeVoice(AbstractApiClass):
    """
        Hume Voice

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name of the voice.
            gender (str): The gender of the voice.
    """

    def __init__(self, client, name=None, gender=None):
        super().__init__(client, None)
        self.name = name
        self.gender = gender
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'gender': repr(self.gender)}
        class_name = "HumeVoice"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'gender': self.gender}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
