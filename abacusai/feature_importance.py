from .return_class import AbstractApiClass


class FeatureImportance(AbstractApiClass):
    """
        Feature importance for a specified model monitor

        Args:
            client (ApiClient): An authenticated API Client instance
            shapFeatureImportance (dict): A feature name, feature importance map for importance determined by shap values on a sample dataset
            permutationFeatureImportance (dict): A feature name, feature importance map for importance determined by permutation importance
            nullFeatureImportance (dict): A feature name, feature importance map for importance determined by null feature importance
            lofoFeatureImportance (dict): A feature name, feature importance map for importance determined by Leave One Feature Out method
    """

    def __init__(self, client, shapFeatureImportance=None, permutationFeatureImportance=None, nullFeatureImportance=None, lofoFeatureImportance=None):
        super().__init__(client, None)
        self.shap_feature_importance = shapFeatureImportance
        self.permutation_feature_importance = permutationFeatureImportance
        self.null_feature_importance = nullFeatureImportance
        self.lofo_feature_importance = lofoFeatureImportance

    def __repr__(self):
        return f"FeatureImportance(shap_feature_importance={repr(self.shap_feature_importance)},\n  permutation_feature_importance={repr(self.permutation_feature_importance)},\n  null_feature_importance={repr(self.null_feature_importance)},\n  lofo_feature_importance={repr(self.lofo_feature_importance)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'shap_feature_importance': self.shap_feature_importance, 'permutation_feature_importance': self.permutation_feature_importance, 'null_feature_importance': self.null_feature_importance, 'lofo_feature_importance': self.lofo_feature_importance}
