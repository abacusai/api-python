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

    def __repr__(self):
        return f"ModelTrainingTypeForDeployment(label={repr(self.label)},\n  value={repr(self.value)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'label': self.label, 'value': self.value}
