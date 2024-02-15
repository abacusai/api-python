from .return_class import AbstractApiClass


class FeatureMapping(AbstractApiClass):
    """
        A description of the data use for a feature

        Args:
            client (ApiClient): An authenticated API Client instance
            featureMapping (str): The mapping of the feature. The possible values will be based on the project's use-case. See the (Use Case Documentation)[https://api.abacus.ai/app/help/useCases] for more details.
            featureName (str): The unique name of the feature.
    """

    def __init__(self, client, featureMapping=None, featureName=None):
        super().__init__(client, None)
        self.feature_mapping = featureMapping
        self.feature_name = featureName
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'feature_mapping': repr(
            self.feature_mapping), f'feature_name': repr(self.feature_name)}
        class_name = "FeatureMapping"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'feature_mapping': self.feature_mapping,
                'feature_name': self.feature_name}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
