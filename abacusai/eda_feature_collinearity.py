from .return_class import AbstractApiClass


class EdaFeatureCollinearity(AbstractApiClass):
    """
        Eda Collinearity of the latest version of the data for a given feature.

        Args:
            client (ApiClient): An authenticated API Client instance
            selectedFeature (str): Selected feature to show the collinearity
            sortedColumnNames (list): Name of all the features in the data sorted in descending order of collinearity value
            featureCollinearity (dict): A dict describing the collinearity between a given feature and all the features in the data
    """

    def __init__(self, client, selectedFeature=None, sortedColumnNames=None, featureCollinearity=None):
        super().__init__(client, None)
        self.selected_feature = selectedFeature
        self.sorted_column_names = sortedColumnNames
        self.feature_collinearity = featureCollinearity

    def __repr__(self):
        repr_dict = {f'selected_feature': repr(self.selected_feature), f'sorted_column_names': repr(
            self.sorted_column_names), f'feature_collinearity': repr(self.feature_collinearity)}
        class_name = "EdaFeatureCollinearity"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'selected_feature': self.selected_feature, 'sorted_column_names':
                self.sorted_column_names, 'feature_collinearity': self.feature_collinearity}
        return {key: value for key, value in resp.items() if value is not None}
