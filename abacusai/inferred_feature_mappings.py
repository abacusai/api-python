from .feature_mapping import FeatureMapping
from .return_class import AbstractApiClass


class InferredFeatureMappings(AbstractApiClass):
    """
        A description of the data use for a feature

        Args:
            client (ApiClient): An authenticated API Client instance
            error (str): Error message if there was an error inferring the feature mappings
            featureMappings (FeatureMapping): The inferred feature mappings
    """

    def __init__(self, client, error=None, featureMappings={}):
        super().__init__(client, None)
        self.error = error
        self.feature_mappings = client._build_class(
            FeatureMapping, featureMappings)

    def __repr__(self):
        return f"InferredFeatureMappings(error={repr(self.error)},\n  feature_mappings={repr(self.feature_mappings)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'error': self.error, 'feature_mappings': self._get_attribute_as_dict(self.feature_mappings)}
