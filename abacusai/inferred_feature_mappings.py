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
        repr_dict = {f'error': repr(
            self.error), f'feature_mappings': repr(self.feature_mappings)}
        class_name = "InferredFeatureMappings"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'error': self.error, 'feature_mappings': self._get_attribute_as_dict(
            self.feature_mappings)}
        return {key: value for key, value in resp.items() if value is not None}
