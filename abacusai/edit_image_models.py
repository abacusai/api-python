from .return_class import AbstractApiClass


class EditImageModels(AbstractApiClass):
    """
        Edit image models

        Args:
            client (ApiClient): An authenticated API Client instance
            models (list): The models available for edit image.
    """

    def __init__(self, client, models=None):
        super().__init__(client, None)
        self.models = models
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'models': repr(self.models)}
        class_name = "EditImageModels"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'models': self.models}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
