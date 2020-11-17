

class Project():
    '''

    '''

    def __init__(self, client, projectId=None, name=None, useCase=None, createdAt=None):
        self.client = client
        self.id = projectId
        self.project_id = projectId
        self.name = name
        self.use_case = useCase
        self.created_at = createdAt

    def __repr__(self):
        return f"Project(project_id={repr(self.project_id)}, name={repr(self.name)}, use_case={repr(self.use_case)}, created_at={repr(self.created_at)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'project_id': self.project_id, 'name': self.name, 'use_case': self.use_case, 'created_at': self.created_at}

    def refresh(self):
        self = self.describe()
        return self

    def describe(self):
        return self.client.describe_project(self.project_id)

    def list_datasets(self):
        return self.client.list_project_datasets(self.project_id)

    def get_schema(self, dataset_id):
        return self.client.get_schema(self.project_id, dataset_id)

    def rename(self, name):
        return self.client.rename_project(self.project_id, name)

    def set_column_data_type(self, dataset_id, column, data_type):
        return self.client.set_column_data_type(self.project_id, dataset_id, column, data_type)

    def set_column_mapping(self, dataset_id, column, column_mapping):
        return self.client.set_column_mapping(self.project_id, dataset_id, column, column_mapping)

    def add_custom_column(self, dataset_id, column, sql):
        return self.client.add_custom_column(self.project_id, dataset_id, column, sql)

    def edit_custom_column(self, dataset_id, column, new_column_name=None, sql=None):
        return self.client.edit_custom_column(self.project_id, dataset_id, column, new_column_name, sql)

    def delete_custom_column(self, dataset_id, column):
        return self.client.delete_custom_column(self.project_id, dataset_id, column)

    def set_dataset_filters(self, dataset_id, filters):
        return self.client.set_project_dataset_filters(self.project_id, dataset_id, filters)

    def validate(self):
        return self.client.validate_project(self.project_id)

    def remove_column_mapping(self, dataset_id, column):
        return self.client.remove_column_mapping(self.project_id, dataset_id, column)

    def delete(self):
        return self.client.delete_project(self.project_id)

    def get_training_config_options(self):
        return self.client.get_training_config_options(self.project_id)

    def train_model(self, name=None, training_config={}, refresh_schedule=None):
        return self.client.train_model(self.project_id, name, training_config, refresh_schedule)

    def list_models(self):
        return self.client.list_models(self.project_id)

    def create_deployment_token(self):
        return self.client.create_deployment_token(self.project_id)

    def list_deployments(self):
        return self.client.list_deployments(self.project_id)

    def list_deployment_tokens(self):
        return self.client.list_deployment_tokens(self.project_id)

    def attach_dataset(self, dataset_id, project_dataset_type):
        return self.client.attach_dataset_to_project(dataset_id, self.project_id, project_dataset_type)

    def remove_dataset(self, dataset_id):
        return self.client.remove_dataset_from_project(dataset_id, self.project_id)
