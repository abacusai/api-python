from .return_class import AbstractApiClass


class FeatureImportance(AbstractApiClass):
    """
        Feature importance for a specified model monitor

        Args:
            client (ApiClient): An authenticated API Client instance
            shapFeatureImportance (dict): A map of feature name to feature importance, determined by Shap values on a sample dataset.
            limeFeatureImportance (dict): A map of feature name to feature importance, determined by Lime contribution values on a sample dataset.
            permutationFeatureImportance (dict): A map of feature name to feature importance, determined by permutation importance.
            nullFeatureImportance (dict): A map of feature name to feature importance, determined by null feature importance.
            lofoFeatureImportance (dict): A map of feature name to feature importance, determined by the Leave One Feature Out method.
            ebmFeatureImportance (dict): A map of feature name to feature importance, determined by an Explainable Boosting Machine.
    """

    def __init__(self, client, shapFeatureImportance=None, limeFeatureImportance=None, permutationFeatureImportance=None, nullFeatureImportance=None, lofoFeatureImportance=None, ebmFeatureImportance=None):
        super().__init__(client, None)
        self.shap_feature_importance = shapFeatureImportance
        self.lime_feature_importance = limeFeatureImportance
        self.permutation_feature_importance = permutationFeatureImportance
        self.null_feature_importance = nullFeatureImportance
        self.lofo_feature_importance = lofoFeatureImportance
        self.ebm_feature_importance = ebmFeatureImportance
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'shap_feature_importance': repr(self.shap_feature_importance), f'lime_feature_importance': repr(self.lime_feature_importance), f'permutation_feature_importance': repr(
            self.permutation_feature_importance), f'null_feature_importance': repr(self.null_feature_importance), f'lofo_feature_importance': repr(self.lofo_feature_importance), f'ebm_feature_importance': repr(self.ebm_feature_importance)}
        class_name = "FeatureImportance"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'shap_feature_importance': self.shap_feature_importance, 'lime_feature_importance': self.lime_feature_importance, 'permutation_feature_importance': self.permutation_feature_importance,
                'null_feature_importance': self.null_feature_importance, 'lofo_feature_importance': self.lofo_feature_importance, 'ebm_feature_importance': self.ebm_feature_importance}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
