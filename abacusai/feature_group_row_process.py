from .return_class import AbstractApiClass


class FeatureGroupRowProcess(AbstractApiClass):
    """
        A feature group row process

        Args:
            client (ApiClient): An authenticated API Client instance
            featureGroupId (str): The ID of the feature group this row that was processed belongs to.
            deploymentId (str): The ID of the deployment that processed this row.
            primaryKeyValue (str): Value of the primary key for this row.
            featureGroupRowProcessId (str): The ID of the feature group row process.
            createdAt (str): The timestamp this feature group row was created in ISO-8601 format.
            updatedAt (str): The timestamp when this feature group row was last updated in ISO-8601 format.
            startedAt (str): The timestamp when this feature group row process was started in ISO-8601 format.
            completedAt (str): The timestamp when this feature group row was completed.
            timeoutAt (str): The time the feature group row process will timeout.
            retriesRemaining (int): The number of retries remaining for this feature group row process.
            totalAttemptsAllowed (int): The total number of attempts allowed for this feature group row process.
            status (str): The status of the feature group row process.
            error (str): The error message if the status is FAILED.
    """

    def __init__(self, client, featureGroupId=None, deploymentId=None, primaryKeyValue=None, featureGroupRowProcessId=None, createdAt=None, updatedAt=None, startedAt=None, completedAt=None, timeoutAt=None, retriesRemaining=None, totalAttemptsAllowed=None, status=None, error=None):
        super().__init__(client, featureGroupRowProcessId)
        self.feature_group_id = featureGroupId
        self.deployment_id = deploymentId
        self.primary_key_value = primaryKeyValue
        self.feature_group_row_process_id = featureGroupRowProcessId
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.started_at = startedAt
        self.completed_at = completedAt
        self.timeout_at = timeoutAt
        self.retries_remaining = retriesRemaining
        self.total_attempts_allowed = totalAttemptsAllowed
        self.status = status
        self.error = error
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'feature_group_id': repr(self.feature_group_id), f'deployment_id': repr(self.deployment_id), f'primary_key_value': repr(self.primary_key_value), f'feature_group_row_process_id': repr(self.feature_group_row_process_id), f'created_at': repr(self.created_at), f'updated_at': repr(
            self.updated_at), f'started_at': repr(self.started_at), f'completed_at': repr(self.completed_at), f'timeout_at': repr(self.timeout_at), f'retries_remaining': repr(self.retries_remaining), f'total_attempts_allowed': repr(self.total_attempts_allowed), f'status': repr(self.status), f'error': repr(self.error)}
        class_name = "FeatureGroupRowProcess"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'feature_group_id': self.feature_group_id, 'deployment_id': self.deployment_id, 'primary_key_value': self.primary_key_value, 'feature_group_row_process_id': self.feature_group_row_process_id, 'created_at': self.created_at, 'updated_at': self.updated_at,
                'started_at': self.started_at, 'completed_at': self.completed_at, 'timeout_at': self.timeout_at, 'retries_remaining': self.retries_remaining, 'total_attempts_allowed': self.total_attempts_allowed, 'status': self.status, 'error': self.error}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def wait_for_process(self, timeout=1200):
        """
        A waiting call until model monitor version is ready.

        Args:
            timeout (int): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        import time
        start_time = time.time()
        feature_group_row_process = None
        while True:
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f'Maximum wait time of {timeout}s exceeded')
            feature_group_row_process = self.client.describe_feature_group_row_process_by_key(
                self.deployment_id, self.primary_key_value)
            if feature_group_row_process.status not in {'PENDING', 'PROCESSING'}:
                break
            time.sleep(5)
        return feature_group_row_process

    def get_status(self):
        """
        Gets the status of the feature group row process.

        Returns:
            str: A string describing the status of the feature group row process
        """
        return self.client.describe_feature_group_row_process_by_key(self.deployment_id, self.primary_key_value).status
