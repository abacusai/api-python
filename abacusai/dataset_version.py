from .return_class import AbstractApiClass


class DatasetVersion(AbstractApiClass):
    """
        A specific version of a dataset

        Args:
            client (ApiClient): An authenticated API Client instance
            datasetVersion (str): The unique identifier of the dataset version.
            status (str): The current status of the dataset version
            datasetId (str): A reference to the Dataset this dataset version belongs to.
            size (int): The size in bytes of the file.
            rowCount (int): Number of rows in the dataset version.
            createdAt (str): The timestamp this dataset version was created.
            error (str): If status is FAILED, this field will be populated with an error.
            invalidRecords (str): 
            incrementalQueriedAt (str): If the dataset version is from an incremental dataset, this is the last entry of timestamp column when the dataset version was created.
    """

    def __init__(self, client, datasetVersion=None, status=None, datasetId=None, size=None, rowCount=None, createdAt=None, error=None, invalidRecords=None, incrementalQueriedAt=None):
        super().__init__(client, datasetVersion)
        self.dataset_version = datasetVersion
        self.status = status
        self.dataset_id = datasetId
        self.size = size
        self.row_count = rowCount
        self.created_at = createdAt
        self.error = error
        self.invalid_records = invalidRecords
        self.incremental_queried_at = incrementalQueriedAt

    def __repr__(self):
        return f"DatasetVersion(dataset_version={repr(self.dataset_version)},\n  status={repr(self.status)},\n  dataset_id={repr(self.dataset_id)},\n  size={repr(self.size)},\n  row_count={repr(self.row_count)},\n  created_at={repr(self.created_at)},\n  error={repr(self.error)},\n  invalid_records={repr(self.invalid_records)},\n  incremental_queried_at={repr(self.incremental_queried_at)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'dataset_version': self.dataset_version, 'status': self.status, 'dataset_id': self.dataset_id, 'size': self.size, 'row_count': self.row_count, 'created_at': self.created_at, 'error': self.error, 'invalid_records': self.invalid_records, 'incremental_queried_at': self.incremental_queried_at}

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            DatasetVersion: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Retrieves a full description of the specified dataset version, with attributes such as its ID, name, source type, etc.

        Args:
            dataset_version (str): The unique ID associated with the dataset version.

        Returns:
            DatasetVersion: The dataset version.
        """
        return self.client.describe_dataset_version(self.dataset_version)

    def wait_for_import(self, timeout=900):
        """
        A waiting call until dataset version is imported.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out. Default value given is 900 milliseconds.
        """
        return self.client._poll(self, {'PENDING', 'IMPORTING'}, timeout=timeout)

    def wait_for_inspection(self, timeout=None):
        """
        A waiting call until dataset version is completely inspected.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.

        """
        return self.client._poll(self, {'PENDING', 'UPLOADING', 'IMPORTING', 'CONVERTING', 'INSPECTING'}, timeout=timeout)

    def get_status(self):
        """
        Gets the status of the dataset version.

        Returns:
            str: A string describing the status of a dataset version (importing, inspecting, complete, etc.).
        """
        return self.describe().status
