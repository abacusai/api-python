from .return_class import AbstractApiClass


class LipSyncGenSettings(AbstractApiClass):
    """
        Lip sync generation settings

        Args:
            client (ApiClient): An authenticated API Client instance
            model (dict): The model settings.
            settings (dict): The settings for each model.
    """

    def __init__(self, client, model=None, settings=None):
        super().__init__(client, None)
        self.model = model
        self.settings = settings
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'model': repr(self.model),
                     f'settings': repr(self.settings)}
        class_name = "LipSyncGenSettings"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'model': self.model, 'settings': self.settings}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
