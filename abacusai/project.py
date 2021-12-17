from .return_class import AbstractApiClass


class Project(AbstractApiClass):
    """
        A project is a container which holds datasets, models and deployments
    """

    def __init__(self, client, projectId=None, name=None, useCase=None, createdAt=None, featureGroupsEnabled=None):
        super().__init__(client, projectId)
        self.project_id = projectId
        self.name = name
        self.use_case = useCase
        self.created_at = createdAt
        self.feature_groups_enabled = featureGroupsEnabled

    def __repr__(self):
        return f"Project(project_id={repr(self.project_id)},\n  name={repr(self.name)},\n  use_case={repr(self.use_case)},\n  created_at={repr(self.created_at)},\n  feature_groups_enabled={repr(self.feature_groups_enabled)})"

    def to_dict(self):
        return {'project_id': self.project_id, 'name': self.name, 'use_case': self.use_case, 'created_at': self.created_at, 'feature_groups_enabled': self.feature_groups_enabled}

    def refresh(self):
        """Calls describe and refreshes the current object's fields"""
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """Returns a description of a project."""
        return self.client.describe_project(self.project_id)

    def list_datasets(self):
        """Retrieves all dataset(s) attached to a specified project. This API returns all attributes of each dataset, such as its name, type, and ID."""
        return self.client.list_project_datasets(self.project_id)

    def get_schema(self, dataset_id):
        """Returns a schema given a specific dataset in a project. The schema of the dataset consists of the columns in the dataset, the data type of the column, and the column's column mapping."""
        return self.client.get_schema(self.project_id, dataset_id)

    def rename(self, name):
        """This method renames a project after it is created."""
        return self.client.rename_project(self.project_id, name)

    def delete(self):
        """Deletes a specified project from your organization."""
        return self.client.delete_project(self.project_id)

    def set_feature_mapping(self, feature_group_id, feature_name, feature_mapping, nested_column_name=None):
        """Set a column's feature mapping. If the column mapping is single-use and already set in another column in this feature group, this call will first remove the other column's mapping and move it to this column."""
        return self.client.set_feature_mapping(self.project_id, feature_group_id, feature_name, feature_mapping, nested_column_name)

    def validate(self):
        """Validates that the specified project has all required feature group types for its use case and that all required feature columns are set."""
        return self.client.validate_project(self.project_id)

    def set_column_data_type(self, dataset_id, column, data_type):
        """Set a dataset's column type."""
        return self.client.set_column_data_type(self.project_id, dataset_id, column, data_type)

    def set_column_mapping(self, dataset_id, column, column_mapping):
        """Set a dataset's column mapping. If the column mapping is single-use and already set in another column in this dataset, this call will first remove the other column's mapping and move it to this column."""
        return self.client.set_column_mapping(self.project_id, dataset_id, column, column_mapping)

    def remove_column_mapping(self, dataset_id, column):
        """Removes a column mapping from a column in the dataset. Returns a list of all columns with their mappings once the change is made."""
        return self.client.remove_column_mapping(self.project_id, dataset_id, column)

    def list_feature_groups(self, filter_feature_group_use=None):
        """List all the feature groups associated with a project"""
        return self.client.list_project_feature_groups(self.project_id, filter_feature_group_use)

    def get_training_config_options(self):
        """Retrieves the full description of the model training configuration options available for the specified project."""
        return self.client.get_training_config_options(self.project_id)

    def train_model(self, name=None, training_config={}, refresh_schedule=None):
        """Trains a model for the specified project."""
        return self.client.train_model(self.project_id, name, training_config, refresh_schedule)

    def create_model_from_python(self, function_source_code, train_function_name, predict_function_name, training_input_tables, name=None):
        """Initializes a new Model from user provided Python code. If a list of input feature groups are supplied,"""
        return self.client.create_model_from_python(self.project_id, function_source_code, train_function_name, predict_function_name, training_input_tables, name)

    def list_models(self):
        """Retrieves the list of models in the specified project."""
        return self.client.list_models(self.project_id)

    def create_model_monitor(self, training_feature_group_id=None, prediction_feature_group_id=None, name=None, refresh_schedule=None):
        """Runs a model monitor for the specified project."""
        return self.client.create_model_monitor(self.project_id, training_feature_group_id, prediction_feature_group_id, name, refresh_schedule)

    def list_model_monitors(self):
        """Retrieves the list of models monitors in the specified project."""
        return self.client.list_model_monitors(self.project_id)

    def create_deployment_token(self):
        """Creates a deployment token for the specified project."""
        return self.client.create_deployment_token(self.project_id)

    def list_deployments(self):
        """Retrieves a list of all deployments in the specified project."""
        return self.client.list_deployments(self.project_id)

    def list_deployment_tokens(self):
        """Retrieves a list of all deployment tokens in the specified project."""
        return self.client.list_deployment_tokens(self.project_id)

    def list_refresh_policies(self, dataset_ids=[], model_ids=[], deployment_ids=[], batch_prediction_ids=[], model_monitor_ids=[]):
        """List the refresh policies for the organization"""
        return self.client.list_refresh_policies(self.project_id, dataset_ids, model_ids, deployment_ids, batch_prediction_ids, model_monitor_ids)

    def list_batch_predictions(self):
        """Retrieves a list for the batch predictions in the project"""
        return self.client.list_batch_predictions(self.project_id)

    def attach_dataset(self, dataset_id, project_dataset_type):
        """
        Attaches dataset to the project.

        Args:
            dataset_id (unique string identifier): A unique identifier for the dataset.
            project_dataset_type (enum of type string): The unique use case specific dataset type that might be required or recommended for the specific use case.

        Returns:
            Schema (object): The schema of the attached dataset.
        """
        return self.client.attach_dataset_to_project(dataset_id, self.project_id, project_dataset_type)

    def remove_dataset(self, dataset_id):
        """
        Removes dataset from the project.

        Args:
            dataset_id (unique string identifier): A unique identifier for the dataset.

        Returns:
            None
        """
        return self.client.remove_dataset_from_project(dataset_id, self.project_id)

    def create_model_from_functions(self, train_function: callable, predict_function: callable, training_input_tables: list = None):
        """
        Creates a model using python.

        Args:
            train_function (callable): The train function is passed.
            predict_function (callable): The prediction function is passed.
            training_input_tables (list, optional): The input tables to be used for training the model. Defaults to None.

        Returns:
            Model (object): The model object.
        """
        return self.client.create_model_from_functions(self.project_id, train_function, predict_function, training_input_tables)
