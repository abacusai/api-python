from .dataset_version import DatasetVersion


class Dataset():
    '''

    '''

    def __init__(self, client, datasetId=None, name=None, sourceType=None, dataSource=None, createdAt=None, refreshSchedules=None, ignoreBefore=None, latestDatasetVersion={}):
        self.client = client
        self.id = datasetId
        self.dataset_id = datasetId
        self.name = name
        self.source_type = sourceType
        self.data_source = dataSource
        self.created_at = createdAt
        self.refresh_schedules = refreshSchedules
        self.ignore_before = ignoreBefore
        self.latest_dataset_version = client._build_class(
            DatasetVersion, latestDatasetVersion)

    def __repr__(self):
        return f"Dataset(dataset_id={repr(self.dataset_id)}, name={repr(self.name)}, source_type={repr(self.source_type)}, data_source={repr(self.data_source)}, created_at={repr(self.created_at)}, refresh_schedules={repr(self.refresh_schedules)}, ignore_before={repr(self.ignore_before)}, latest_dataset_version={repr(self.latest_dataset_version)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'dataset_id': self.dataset_id, 'name': self.name, 'source_type': self.source_type, 'data_source': self.data_source, 'created_at': self.created_at, 'refresh_schedules': self.refresh_schedules, 'ignore_before': self.ignore_before, 'latest_dataset_version': self.latest_dataset_version.to_dict() if self.latest_dataset_version else None}

    def create_version(self, location=None, file_format=None):
        return self.client.create_dataset_version(self.dataset_id, location, file_format)

    def create_version_from_local_file(self, file_format=None):
        return self.client.create_dataset_version_from_local_file(self.dataset_id, file_format)

    def set_ignore_before(self, timestamp=None):
        return self.client.set_ignore_before(self.dataset_id, timestamp)

    def refresh(self):
        self = self.describe()
        return self

    def describe(self):
        return self.client.describe_dataset(self.dataset_id)

    def list_versions(self):
        return self.client.list_dataset_versions(self.dataset_id)

    def attach_to_project(self, project_id, dataset_type):
        return self.client.attach_dataset_to_project(self.dataset_id, project_id, dataset_type)

    def remove_from_project(self, project_id):
        return self.client.remove_dataset_from_project(self.dataset_id, project_id)

    def rename(self, name):
        return self.client.rename_dataset(self.dataset_id, name)

    def delete(self):
        return self.client.delete_dataset(self.dataset_id)

    def wait_for_import(self, timeout=900):
        return self.client._poll(self, {'PENDING', 'IMPORTING'}, timeout=timeout)

    def wait_for_inspection(self, timeout=None):
        return self.client._poll(self, {'PENDING', 'UPLOADING', 'IMPORTING', 'CONVERTING', 'INSPECTING'}, timeout=timeout)

    def get_status(self):
        return self.describe().latest_dataset_version.status
