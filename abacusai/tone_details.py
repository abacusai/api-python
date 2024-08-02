from .return_class import AbstractApiClass


class ToneDetails(AbstractApiClass):
    """
        Tone details for audio

        Args:
            client (ApiClient): An authenticated API Client instance
            voiceId (str): The voice id
            name (str): The name
            gender (str): The gender
            language (str): The language
            age (str): The age
            accent (str): The accent
            useCase (str): The use case
            description (str): The description
    """

    def __init__(self, client, voiceId=None, name=None, gender=None, language=None, age=None, accent=None, useCase=None, description=None):
        super().__init__(client, None)
        self.voice_id = voiceId
        self.name = name
        self.gender = gender
        self.language = language
        self.age = age
        self.accent = accent
        self.use_case = useCase
        self.description = description
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'voice_id': repr(self.voice_id), f'name': repr(self.name), f'gender': repr(self.gender), f'language': repr(
            self.language), f'age': repr(self.age), f'accent': repr(self.accent), f'use_case': repr(self.use_case), f'description': repr(self.description)}
        class_name = "ToneDetails"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'voice_id': self.voice_id, 'name': self.name, 'gender': self.gender, 'language': self.language,
                'age': self.age, 'accent': self.accent, 'use_case': self.use_case, 'description': self.description}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
