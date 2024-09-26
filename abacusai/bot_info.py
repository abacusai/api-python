from .return_class import AbstractApiClass


class BotInfo(AbstractApiClass):
    """
        Information about an external application and LLM.

        Args:
            client (ApiClient): An authenticated API Client instance
            externalApplicationId (str): The external application ID.
    """

    def __init__(self, client, externalApplicationId=None):
        super().__init__(client, None)
        self.external_application_id = externalApplicationId
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'external_application_id': repr(
            self.external_application_id)}
        class_name = "BotInfo"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'external_application_id': self.external_application_id}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
