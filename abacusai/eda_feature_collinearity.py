from .return_class import AbstractApiClass


class EdaFeatureCollinearity(AbstractApiClass):
    """
        Eda Collinearity of the latest version of the data for a given feature.

        Args:
            client (ApiClient): An authenticated API Client instance
            selectedFeature (str): Selected feature to show the collinearity
            sortedColumnNames (list): Name of all the features in the data sorted in descending order of collinearity value
            featureCollinearity (CollinearityRecord): A sorted List describing the collinearity between a given feature and all the features in the data
    """

    def __init__(self, client, selectedFeature=None, sortedColumnNames=None, featureCollinearity={}):
        super().__init__(client, None)
        self.selected_feature = selectedFeature
        self.sorted_column_names = sortedColumnNames
        self.feature_collinearity = client._build_class(
            CollinearityRecord, featureCollinearity)

    def __repr__(self):
        return f"EdaFeatureCollinearity(selected_feature={repr(self.selected_feature)},\n  sorted_column_names={repr(self.sorted_column_names)},\n  feature_collinearity={repr(self.feature_collinearity)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'selected_feature': self.selected_feature, 'sorted_column_names': self.sorted_column_names, 'feature_collinearity': self._get_attribute_as_dict(self.feature_collinearity)}
