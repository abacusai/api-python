

class Project():
    '''
        A project is a container which holds datasets, models and deployments
    '''

    def __init__(self, client, projectId=None, name=None, useCase=None, createdAt=None, featureGroupsEnabled=None):
        self.client = client
        self.id = projectId
        self.project_id = projectId
        self.name = name
        self.use_case = useCase
        self.created_at = createdAt
        self.feature_groups_enabled = featureGroupsEnabled

    def __repr__(self):
        return f"Project(project_id={repr(self.project_id)}, name={repr(self.name)}, use_case={repr(self.use_case)}, created_at={repr(self.created_at)}, feature_groups_enabled={repr(self.feature_groups_enabled)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'project_id': self.project_id, 'name': self.name, 'use_case': self.use_case, 'created_at': self.created_at, 'feature_groups_enabled': self.feature_groups_enabled}

    def refresh(self):
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        return self.client.describe_project(self.project_id)

    def list_datasets(self):
        return self.client.list_project_datasets(self.project_id)

    def get_schema(self, dataset_id):
        return self.client.get_schema(self.project_id, dataset_id)

    def rename(self, name):
        return self.client.rename_project(self.project_id, name)

    def delete(self):
        return self.client.delete_project(self.project_id)

    def set_feature_mapping(self, feature_group_id, feature_name, feature_mapping):
        return self.client.set_feature_mapping(self.project_id, feature_group_id, feature_name, feature_mapping)

    def validate(self):
        return self.client.validate_project(self.project_id)

    def set_column_data_type(self, dataset_id, column, data_type):
        return self.client.set_column_data_type(self.project_id, dataset_id, column, data_type)

    def set_column_mapping(self, dataset_id, column, column_mapping):
        return self.client.set_column_mapping(self.project_id, dataset_id, column, column_mapping)

    def remove_column_mapping(self, dataset_id, column):
        return self.client.remove_column_mapping(self.project_id, dataset_id, column)

    def list_feature_groups(self):
        return self.client.list_project_feature_groups(self.project_id)

    def get_training_config_options(self):
        return self.client.get_training_config_options(self.project_id)

    def train_model(self, name=None, training_config={}, refresh_schedule=None):
        return self.client.train_model(self.project_id, name, training_config, refresh_schedule)

    def create_python_model(self, function_source_code, train_function_name, predict_function_name, training_input_tables=[], name=None):
        return self.client.create_python_model(self.project_id, function_source_code, train_function_name, predict_function_name, training_input_tables, name)

    def list_models(self):
        return self.client.list_models(self.project_id)

    def create_deployment_token(self):
        return self.client.create_deployment_token(self.project_id)

    def list_deployments(self):
        return self.client.list_deployments(self.project_id)

    def list_deployment_tokens(self):
        return self.client.list_deployment_tokens(self.project_id)

    def list_batch_predictions(self):
        return self.client.list_batch_predictions(self.project_id)

    def attach_dataset(self, dataset_id, project_dataset_type):
        return self.client.attach_dataset_to_project(dataset_id, self.project_id, project_dataset_type)

    def remove_dataset(self, dataset_id):
        return self.client.remove_dataset_from_project(dataset_id, self.project_id)
