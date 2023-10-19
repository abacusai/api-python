from .return_class import AbstractApiClass


class HoldoutAnalysis(AbstractApiClass):
    """
        A holdout analysis object.

        Args:
            client (ApiClient): An authenticated API Client instance
            holdoutAnalysisId (str): The unique identifier of the holdout analysis.
            name (str): The name of the holdout analysis.
            featureGroupIds (list[str]): The feature group ids associated with the holdout analysis.
            modelId (str): The model id associated with the holdout analysis.
            modelName (str): The model name associated with the holdout analysis.
    """

    def __init__(self, client, holdoutAnalysisId=None, name=None, featureGroupIds=None, modelId=None, modelName=None):
        super().__init__(client, holdoutAnalysisId)
        self.holdout_analysis_id = holdoutAnalysisId
        self.name = name
        self.feature_group_ids = featureGroupIds
        self.model_id = modelId
        self.model_name = modelName

    def __repr__(self):
        repr_dict = {f'holdout_analysis_id': repr(self.holdout_analysis_id), f'name': repr(self.name), f'feature_group_ids': repr(
            self.feature_group_ids), f'model_id': repr(self.model_id), f'model_name': repr(self.model_name)}
        class_name = "HoldoutAnalysis"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'holdout_analysis_id': self.holdout_analysis_id, 'name': self.name,
                'feature_group_ids': self.feature_group_ids, 'model_id': self.model_id, 'model_name': self.model_name}
        return {key: value for key, value in resp.items() if value is not None}

    def rerun(self, model_version: str = None, algorithm: str = None):
        """
        Rerun a holdout analysis. A different model version and algorithm can be specified which should be under the same model.

        Args:
            model_version (str): (optional) Version of the model to use for the holdout analysis
            algorithm (str): (optional) ID of algorithm to use for the holdout analysis

        Returns:
            HoldoutAnalysisVersion: The created holdout analysis version
        """
        return self.client.rerun_holdout_analysis(self.holdout_analysis_id, model_version, algorithm)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            HoldoutAnalysis: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Get a holdout analysis.

        Args:
            holdout_analysis_id (str): ID of the holdout analysis to get

        Returns:
            HoldoutAnalysis: The holdout analysis
        """
        return self.client.describe_holdout_analysis(self.holdout_analysis_id)

    def list_versions(self):
        """
        List holdout analysis versions for a holdout analysis.

        Args:
            holdout_analysis_id (str): ID of the holdout analysis to list holdout analysis versions for

        Returns:
            list[HoldoutAnalysisVersion]: The holdout analysis versions
        """
        return self.client.list_holdout_analysis_versions(self.holdout_analysis_id)
