from .feature_group import FeatureGroup
from .feature_group_version import FeatureGroupVersion
from .return_class import AbstractApiClass


class OrganizationSearchResult(AbstractApiClass):
    """
        A search result object which contains the retrieved artifact and its relevance score

        Args:
            client (ApiClient): An authenticated API Client instance
            score (float): The relevance score of the search result.
            featureGroupContext (str): The rendered context for the feature group that can be used in prompts
            featureGroup (FeatureGroup): The feature group object retrieved through search.
            featureGroupVersion (FeatureGroupVersion): The feature group version object retrieved through search.
    """

    def __init__(self, client, score=None, featureGroupContext=None, featureGroup={}, featureGroupVersion={}):
        super().__init__(client, None)
        self.score = score
        self.feature_group_context = featureGroupContext
        self.feature_group = client._build_class(FeatureGroup, featureGroup)
        self.feature_group_version = client._build_class(
            FeatureGroupVersion, featureGroupVersion)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'score': repr(self.score), f'feature_group_context': repr(self.feature_group_context), f'feature_group': repr(
            self.feature_group), f'feature_group_version': repr(self.feature_group_version)}
        class_name = "OrganizationSearchResult"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'score': self.score, 'feature_group_context': self.feature_group_context, 'feature_group': self._get_attribute_as_dict(
            self.feature_group), 'feature_group_version': self._get_attribute_as_dict(self.feature_group_version)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
