from .code_source import CodeSource
from .return_class import AbstractApiClass


class PredictionOperatorVersion(AbstractApiClass):
    """
        A prediction operator version.

        Args:
            client (ApiClient): An authenticated API Client instance
            predictionOperatorId (str): The unique identifier of the prediction operator.
            predictionOperatorVersion (str): The unique identifier of the prediction operator version.
            createdAt (str): Date and time at which the prediction operator was created.
            updatedAt (str): Date and time at which the prediction operator was updated.
            sourceCode (str): Python code used to make the prediction operator.
            memory (int): Memory in GB specified for the prediction operator version.
            useGpu (bool): Whether this prediction operator version is using gpu.
            featureGroupIds (list): A list of Feature Group IDs used for initializing.
            featureGroupVersions (list): A list of Feature Group version IDs used for initializing.
            codeSource (CodeSource): If a python model, information on the source code.
    """

    def __init__(self, client, predictionOperatorId=None, predictionOperatorVersion=None, createdAt=None, updatedAt=None, sourceCode=None, memory=None, useGpu=None, featureGroupIds=None, featureGroupVersions=None, codeSource={}):
        super().__init__(client, predictionOperatorVersion)
        self.prediction_operator_id = predictionOperatorId
        self.prediction_operator_version = predictionOperatorVersion
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.source_code = sourceCode
        self.memory = memory
        self.use_gpu = useGpu
        self.feature_group_ids = featureGroupIds
        self.feature_group_versions = featureGroupVersions
        self.code_source = client._build_class(CodeSource, codeSource)

    def __repr__(self):
        return f"PredictionOperatorVersion(prediction_operator_id={repr(self.prediction_operator_id)},\n  prediction_operator_version={repr(self.prediction_operator_version)},\n  created_at={repr(self.created_at)},\n  updated_at={repr(self.updated_at)},\n  source_code={repr(self.source_code)},\n  memory={repr(self.memory)},\n  use_gpu={repr(self.use_gpu)},\n  feature_group_ids={repr(self.feature_group_ids)},\n  feature_group_versions={repr(self.feature_group_versions)},\n  code_source={repr(self.code_source)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'prediction_operator_id': self.prediction_operator_id, 'prediction_operator_version': self.prediction_operator_version, 'created_at': self.created_at, 'updated_at': self.updated_at, 'source_code': self.source_code, 'memory': self.memory, 'use_gpu': self.use_gpu, 'feature_group_ids': self.feature_group_ids, 'feature_group_versions': self.feature_group_versions, 'code_source': self._get_attribute_as_dict(self.code_source)}

    def delete(self):
        """
        Delete a prediction operator version.

        Args:
            prediction_operator_version (str): The unique ID of the prediction operator version.

        Returns:
            PredictionOperatorVersion: 
        """
        return self.client.delete_prediction_operator_version(self.prediction_operator_version)
