import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from .return_class import AbstractApiClass


class ExecuteFgOperation(AbstractApiClass):
    """
        The result of executing a SQL query

        Args:
            client (ApiClient): An authenticated API Client instance
            executeFgOperationRunId (str): The run id of the operation
            status (str): The status of the operation
            error (str): The error message if the operation failed
    """

    def __init__(self, client, executeFgOperationRunId=None, status=None, error=None):
        super().__init__(client, None)
        self.execute_fg_operation_run_id = executeFgOperationRunId
        self.status = status
        self.error = error

    def __repr__(self):
        return f"ExecuteFgOperation(execute_fg_operation_run_id={repr(self.execute_fg_operation_run_id)},\n  status={repr(self.status)},\n  error={repr(self.error)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'execute_fg_operation_run_id': self.execute_fg_operation_run_id, 'status': self.status, 'error': self.error}

    def wait_for_results(self, timeout=3600):
        """
        A waiting call until query is executed.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'GENERATING'}, timeout=timeout)

    def wait_for_execution(self, timeout=3600):
        """
        A waiting call until query is executed.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.wait_for_results(timeout)

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
        return self.client.describe_async_feature_group_operation(self.execute_fg_operation_run_id)

    # internal call
    def _download_avro_file(self, file_part, tmp_dir):
        offset = 0
        part_path = os.path.join(tmp_dir, f'{file_part}.avro')
        with open(part_path, 'wb') as file:
            while True:
                with self.client._call_api('downloadExecuteFeatureGroupOperationResultPartChunk', 'GET', query_params={'executeFgOperationRunId': self.execute_fg_operation_run_id, 'part': file_part, 'offset': offset}, streamable_response=True) as chunk:
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
        import fastavro
        import pandas as pd

        file_parts = range(self.client._call_api('getExecuteFeatureGroupOperationResultPartCount',
                           'POST', query_params={'executeFgOperationRunId': self.execute_fg_operation_run_id}))
        data_df = pd.DataFrame()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tempfile.TemporaryDirectory() as tmp_dir:
                df_parts = []
                file_futures = [executor.submit(
                    self._download_avro_file, file_part, tmp_dir) for file_part in file_parts]
                for future in file_futures:
                    part_path = future.result()
                    with open(part_path, 'rb') as part_data:
                        reader = fastavro.reader(part_data)
                        df_parts.append(pd.DataFrame([r for r in reader]))
                data_df = pd.concat(df_parts, ignore_index=True)
            for col in data_df.columns:
                if pd.core.dtypes.common.is_datetime64_ns_dtype(data_df[col]):
                    data_df[col] = data_df[col].dt.tz_localize(None)
        return data_df
