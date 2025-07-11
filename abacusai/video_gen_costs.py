from .return_class import AbstractApiClass


class VideoGenCosts(AbstractApiClass):
    """
        The most expensive price for each video gen model in credits

        Args:
            client (ApiClient): An authenticated API Client instance
            modelCosts (dict): The costs of the video gen models in credits
            expensiveModels (list): The list of video gen models that are expensive
            warningMessages (dict): The warning messages for certain video gen models
    """

    def __init__(self, client, modelCosts=None, expensiveModels=None, warningMessages=None):
        super().__init__(client, None)
        self.model_costs = modelCosts
        self.expensive_models = expensiveModels
        self.warning_messages = warningMessages
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'model_costs': repr(self.model_costs), f'expensive_models': repr(
            self.expensive_models), f'warning_messages': repr(self.warning_messages)}
        class_name = "VideoGenCosts"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'model_costs': self.model_costs, 'expensive_models': self.expensive_models,
                'warning_messages': self.warning_messages}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
