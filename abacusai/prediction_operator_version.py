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
            status (str): The current status of the prediction operator version.
            error (str): The error message if the status failed.
            codeSource (CodeSource): If a python model, information on the source code.
    """

    def __init__(self, client, predictionOperatorId=None, predictionOperatorVersion=None, createdAt=None, updatedAt=None, sourceCode=None, memory=None, useGpu=None, featureGroupIds=None, featureGroupVersions=None, status=None, error=None, codeSource={}):
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
        self.status = status
        self.error = error
        self.code_source = client._build_class(CodeSource, codeSource)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'prediction_operator_id': repr(self.prediction_operator_id), f'prediction_operator_version': repr(self.prediction_operator_version), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at), f'source_code': repr(self.source_code), f'memory': repr(
            self.memory), f'use_gpu': repr(self.use_gpu), f'feature_group_ids': repr(self.feature_group_ids), f'feature_group_versions': repr(self.feature_group_versions), f'status': repr(self.status), f'error': repr(self.error), f'code_source': repr(self.code_source)}
        class_name = "PredictionOperatorVersion"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'prediction_operator_id': self.prediction_operator_id, 'prediction_operator_version': self.prediction_operator_version, 'created_at': self.created_at, 'updated_at': self.updated_at, 'source_code': self.source_code, 'memory': self.memory,
                'use_gpu': self.use_gpu, 'feature_group_ids': self.feature_group_ids, 'feature_group_versions': self.feature_group_versions, 'status': self.status, 'error': self.error, 'code_source': self._get_attribute_as_dict(self.code_source)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def delete(self):
        """
        Delete a prediction operator version.

        Args:
            prediction_operator_version (str): The unique ID of the prediction operator version.
        """
        return self.client.delete_prediction_operator_version(self.prediction_operator_version)
