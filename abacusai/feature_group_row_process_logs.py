from .return_class import AbstractApiClass


class FeatureGroupRowProcessLogs(AbstractApiClass):
    """
        Logs for the feature group row process.

        Args:
            client (ApiClient): An authenticated API Client instance
            logs (str): The logs for both stdout and stderr of the step
            featureGroupId (str): The ID of the feature group this row that was processed belongs to.
            deploymentId (str): The ID of the deployment that processed this row.
            primaryKeyValue (str): Value of the primary key for this row.
            featureGroupRowProcessId (str): The ID of the feature group row process.
    """

    def __init__(self, client, logs=None, featureGroupId=None, deploymentId=None, primaryKeyValue=None, featureGroupRowProcessId=None):
        super().__init__(client, None)
        self.logs = logs
        self.feature_group_id = featureGroupId
        self.deployment_id = deploymentId
        self.primary_key_value = primaryKeyValue
        self.feature_group_row_process_id = featureGroupRowProcessId
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'logs': repr(self.logs), f'feature_group_id': repr(self.feature_group_id), f'deployment_id': repr(
            self.deployment_id), f'primary_key_value': repr(self.primary_key_value), f'feature_group_row_process_id': repr(self.feature_group_row_process_id)}
        class_name = "FeatureGroupRowProcessLogs"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'logs': self.logs, 'feature_group_id': self.feature_group_id, 'deployment_id': self.deployment_id,
                'primary_key_value': self.primary_key_value, 'feature_group_row_process_id': self.feature_group_row_process_id}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
