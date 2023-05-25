from .feature_group import FeatureGroup
from .feature_group_version import FeatureGroupVersion
from .return_class import AbstractApiClass


class OrganizationSearchResult(AbstractApiClass):
    """
        A search result object which contains the retrieved artifact and its relevance score

        Args:
            client (ApiClient): An authenticated API Client instance
            score (float): The relevance score of the search result.
            featureGroup (FeatureGroup): The feature group object retrieved through search.
            featureGroupVersion (FeatureGroupVersion): The feature group version object retrieved through search.
    """

    def __init__(self, client, score=None, featureGroup={}, featureGroupVersion={}):
        super().__init__(client, None)
        self.score = score
        self.feature_group = client._build_class(FeatureGroup, featureGroup)
        self.feature_group_version = client._build_class(
            FeatureGroupVersion, featureGroupVersion)

    def __repr__(self):
        return f"OrganizationSearchResult(score={repr(self.score)},\n  feature_group={repr(self.feature_group)},\n  feature_group_version={repr(self.feature_group_version)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'score': self.score, 'feature_group': self._get_attribute_as_dict(self.feature_group), 'feature_group_version': self._get_attribute_as_dict(self.feature_group_version)}
