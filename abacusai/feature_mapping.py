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

    def __repr__(self):
        return f"FeatureMapping(feature_mapping={repr(self.feature_mapping)},\n  feature_name={repr(self.feature_name)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'feature_mapping': self.feature_mapping, 'feature_name': self.feature_name}
