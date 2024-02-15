from .return_class import AbstractApiClass


class ModelTrainingTypeForDeployment(AbstractApiClass):
    """
        Model training types for deployment.

        Args:
            client (ApiClient): An authenticated API Client instance
            label (str): Labels to show to users in deployment UI
            value (str): Value to use on backend for deployment API call
    """

    def __init__(self, client, label=None, value=None):
        super().__init__(client, None)
        self.label = label
        self.value = value
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'label': repr(self.label), f'value': repr(self.value)}
        class_name = "ModelTrainingTypeForDeployment"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'label': self.label, 'value': self.value}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
