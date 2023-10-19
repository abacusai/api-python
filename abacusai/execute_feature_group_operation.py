import os

from .return_class import AbstractApiClass


class ExecuteFeatureGroupOperation(AbstractApiClass):
    """
        The result of executing a SQL query

        Args:
            client (ApiClient): An authenticated API Client instance
            featureGroupOperationRunId (str): The run id of the operation
            status (str): The status of the operation
            error (str): The error message if the operation failed
    """

    def __init__(self, client, featureGroupOperationRunId=None, status=None, error=None):
        super().__init__(client, None)
        self.feature_group_operation_run_id = featureGroupOperationRunId
        self.status = status
        self.error = error

    def __repr__(self):
        repr_dict = {f'feature_group_operation_run_id': repr(
            self.feature_group_operation_run_id), f'status': repr(self.status), f'error': repr(self.error)}
        class_name = "ExecuteFeatureGroupOperation"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'feature_group_operation_run_id': self.feature_group_operation_run_id,
                'status': self.status, 'error': self.error}
        return {key: value for key, value in resp.items() if value is not None}

    def wait_for_results(self, timeout=3600, delay=2):
        """
        A waiting call until query is executed.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'GENERATING'}, timeout=timeout, delay=delay)

    def wait_for_execution(self, timeout=3600, delay=2):
        """
        A waiting call until query is executed.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.wait_for_results(timeout, delay=delay)

    def get_status(self):
        """
        Gets the status of the query execution

        Returns:
            str: A string describing the status of a query execution (pending, complete, etc.).
        """
        return self.describe().status

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            DatasetVersion: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        return self.client.describe_async_feature_group_operation(self.feature_group_operation_run_id)

    def _download_avro_file(self, file_part, tmp_dir, part_index):
        from .api_client_utils import try_abacus_internal_copy

        part_path = os.path.join(tmp_dir, f'{part_index}.avro')
        if try_abacus_internal_copy(file_part, part_path):
            return part_path

        offset = 0
        with open(part_path, 'wb') as file:
            while True:
                with self.client._call_api('downloadExecuteFeatureGroupOperationResultPartChunk', 'GET', query_params={'featureGroupOperationRunId': self.feature_group_operation_run_id, 'part': part_index, 'offset': offset}, streamable_response=True) as chunk:
                    bytes_written = file.write(chunk.read())
                if not bytes_written:
                    break
                offset += bytes_written
        return part_path

    def load_as_pandas(self, max_workers=10):
        """
        Loads the result data into a pandas dataframe

        Args:
            max_workers (int, optional): The number of threads.

        Returns:
            DataFrame: A pandas dataframe displaying the data from execution.
        """

        from .api_client_utils import load_as_pandas_from_avro_files

        file_parts = self.client._call_api(
            '_getExecuteFeatureGroupOperationResultParts', 'POST', query_params={'featureGroupOperationRunId': self.feature_group_operation_run_id})
        return load_as_pandas_from_avro_files(file_parts, self._download_avro_file, max_workers=max_workers)
