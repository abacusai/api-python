from .return_class import AbstractApiClass


class Project(AbstractApiClass):
    """
        A project is a container which holds datasets, models and deployments

        Args:
            client (ApiClient): An authenticated API Client instance
            projectId (str): The ID of the project.
            name (str): The name of the project.
            useCase (str): The  Use Case associated with the project.
            createdAt (str): The date and time when the project was created.
            featureGroupsEnabled (bool): Project uses feature groups instead of datasets.
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
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'project_id': self.project_id, 'name': self.name, 'use_case': self.use_case, 'created_at': self.created_at, 'feature_groups_enabled': self.feature_groups_enabled}

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            Project: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Returns a description of a project.

        Args:
            project_id (str): The unique project ID

        Returns:
            Project: The project description is returned.
        """
        return self.client.describe_project(self.project_id)

    def list_datasets(self):
        """
        Retrieves all dataset(s) attached to a specified project. This API returns all attributes of each dataset, such as its name, type, and ID.

        Args:
            project_id (str): The unique ID associated with the project.

        Returns:
            ProjectDataset: An array representing all of the datasets attached to the project.
        """
        return self.client.list_project_datasets(self.project_id)

    def get_schema(self, dataset_id: str):
        """
        [DEPRECATED] Returns a schema given a specific dataset in a project. The schema of the dataset consists of the columns in the dataset, the data type of the column, and the column's column mapping.

        Args:
            dataset_id (str): The unique ID associated with the dataset.

        Returns:
            Schema: An array of objects for each column in the specified dataset.
        """
        return self.client.get_schema(self.project_id, dataset_id)

    def rename(self, name: str):
        """
        This method renames a project after it is created.

        Args:
            name (str): The new name for the project.
        """
        return self.client.rename_project(self.project_id, name)

    def delete(self):
        """
        Deletes a specified project from your organization.

        This method deletes the project, trained models and deployments in the specified project. The datasets attached to the specified project remain available for use with other projects in the organization.

        This method will not delete a project that contains active deployments. Be sure to stop all active deployments before you use the delete option.

        Note: All projects, models, and deployments cannot be recovered once they are deleted.


        Args:
            project_id (str): The unique ID of the project to delete.
        """
        return self.client.delete_project(self.project_id)

    def set_feature_mapping(self, feature_group_id: str, feature_name: str, feature_mapping: str, nested_column_name: str = None):
        """
        Set a column's feature mapping. If the column mapping is single-use and already set in another column in this feature group, this call will first remove the other column's mapping and move it to this column.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            feature_name (str): The name of the feature.
            feature_mapping (str): The mapping of the feature in the feature group.
            nested_column_name (str): The name of the nested column.

        Returns:
            Feature: A list of objects that describes the resulting feature group's schema after the feature's featureMapping is set.
        """
        return self.client.set_feature_mapping(self.project_id, feature_group_id, feature_name, feature_mapping, nested_column_name)

    def validate(self, feature_group_ids: list = None):
        """
        Validates that the specified project has all required feature group types for its use case and that all required feature columns are set.

        Args:
            feature_group_ids (list): The feature group IDS to validate

        Returns:
            ProjectValidation: The project validation. If the specified project is missing required columns or feature groups, the response includes an array of objects for each missing required feature group and the missing required features in each feature group.
        """
        return self.client.validate_project(self.project_id, feature_group_ids)

    def set_column_data_type(self, dataset_id: str, column: str, data_type: str):
        """
        Set a dataset's column type.

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            column (str): The name of the column.
            data_type (str): The type of the data in the column.  CATEGORICAL,  CATEGORICAL_LIST,  NUMERICAL,  TIMESTAMP,  TEXT,  EMAIL,  LABEL_LIST,  JSON,  OBJECT_REFERENCE Refer to the (guide on feature types)[https://api.abacus.ai/app/help/class/FeatureType] for more information. Note: Some ColumnMappings will restrict the options or explicity set the DataType.

        Returns:
            Schema: A list of objects that describes the resulting dataset's schema after the column's dataType is set.
        """
        return self.client.set_column_data_type(self.project_id, dataset_id, column, data_type)

    def set_column_mapping(self, dataset_id: str, column: str, column_mapping: str):
        """
        Set a dataset's column mapping. If the column mapping is single-use and already set in another column in this dataset, this call will first remove the other column's mapping and move it to this column.

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            column (str): The name of the column.
            column_mapping (str): The mapping of the column in the dataset. See a list of columns mapping enums here.

        Returns:
            Schema: A list of columns that describes the resulting dataset's schema after the column's columnMapping is set.
        """
        return self.client.set_column_mapping(self.project_id, dataset_id, column, column_mapping)

    def remove_column_mapping(self, dataset_id: str, column: str):
        """
        Removes a column mapping from a column in the dataset. Returns a list of all columns with their mappings once the change is made.

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            column (str): The name of the column.

        Returns:
            Schema: A list of objects that describes the resulting dataset's schema after the column's columnMapping is set.
        """
        return self.client.remove_column_mapping(self.project_id, dataset_id, column)

    def list_feature_groups(self, filter_feature_group_use: str = None):
        """
        List all the feature groups associated with a project

        Args:
            filter_feature_group_use (str): The feature group use filter, when given as an argument, only allows feature groups in this project to be returned if they are of the given use.  DATA_WRANGLING,  TRAINING_INPUT,  BATCH_PREDICTION_INPUT,  BATCH_PREDICTION_OUTPUT

        Returns:
            FeatureGroup: All the Feature Groups in the Organization
        """
        return self.client.list_project_feature_groups(self.project_id, filter_feature_group_use)

    def get_training_config_options(self, feature_group_ids: list = None):
        """
        Retrieves the full description of the model training configuration options available for the specified project.

        The configuration options available are determined by the use case associated with the specified project. Refer to the (Use Case Documentation)[https://api.abacus.ai/app/help/useCases] for more information on use cases and use case specific configuration options.


        Args:
            feature_group_ids (list): The feature group IDs to be used for training

        Returns:
            TrainingConfigOptions: An array of options that can be specified when training a model in this project.
        """
        return self.client.get_training_config_options(self.project_id, feature_group_ids)

    def train_model(self, name: str = None, training_config: dict = None, feature_group_ids: list = None, refresh_schedule: str = None):
        """
        Trains a model for the specified project.

        Use this method to train a model in this project. This method supports user-specified training configurations defined in the getTrainingConfigOptions method.


        Args:
            name (str): The name you want your model to have. Defaults to "<Project Name> Model".
            training_config (dict): The training config key/value pairs used to train this model.
            feature_group_ids (list): List of feature group ids provided by the user to train the model on.
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically retrain the created model.

        Returns:
            Model: The new model which is being trained.
        """
        return self.client.train_model(self.project_id, name, training_config, feature_group_ids, refresh_schedule)

    def create_model_from_python(self, function_source_code: str, train_function_name: str, predict_function_name: str, training_input_tables: list, name: str = None, cpu_size: str = None, memory: int = None, training_config: dict = None, exclusive_run: bool = False, package_requirements: dict = None):
        """
        Initializes a new Model from user provided Python code. If a list of input feature groups are supplied,

        we will provide as arguments to the train and predict functions with the materialized feature groups for those
        input feature groups.

        This method expects `functionSourceCode` to be a valid language source file which contains the functions named
        `trainFunctionName` and `predictFunctionName`. `trainFunctionName` returns the ModelVersion that is the result of
        training the model using `trainFunctionName` and `predictFunctionName` has no well defined return type,
        as it returns the prediction made by the `predictFunctionName`, which can be anything


        Args:
            function_source_code (str): Contents of a valid python source code file. The source code should contain the functions named trainFunctionName and predictFunctionName. A list of allowed import and system libraries for each language is specified in the user functions documentation section.
            train_function_name (str): Name of the function found in the source code that will be executed to train the model. It is not executed when this function is run.
            predict_function_name (str): Name of the function found in the source code that will be executed run predictions through model. It is not executed when this function is run.
            training_input_tables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            name (str): The name you want your model to have. Defaults to "<Project Name> Model"
            cpu_size (str): Size of the cpu for the model training function
            memory (int): Memory (in GB) for the model training function
            training_config (dict): Training configuration
            exclusive_run (bool): Decides if this model will be run exclusively OR along with other Abacus.ai algorithms
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency

        Returns:
            Model: The new model, which has not been trained.
        """
        return self.client.create_model_from_python(self.project_id, function_source_code, train_function_name, predict_function_name, training_input_tables, name, cpu_size, memory, training_config, exclusive_run, package_requirements)

    def create_model_from_zip(self, train_function_name: str, predict_function_name: str, train_module_name: str, predict_module_name: str, training_input_tables: list, name: str = None, cpu_size: str = None, memory: int = None, package_requirements: dict = None):
        """
        Initializes a new Model from a user provided zip file containing Python code. If a list of input feature groups are supplied,

        we will provide as arguments to the train and predict functions with the materialized feature groups for those
        input feature groups.

        This method expects `trainModuleName` and `predictModuleName` to be valid language source files which contains the functions named
        `trainFunctionName` and `predictFunctionName`, respectively. `trainFunctionName` returns the ModelVersion that is the result of
        training the model using `trainFunctionName` and `predictFunctionName` has no well defined return type,
        as it returns the prediction made by the `predictFunctionName`, which can be anything


        Args:
            train_function_name (str): Name of the function found in train module that will be executed to train the model. It is not executed when this function is run.
            predict_function_name (str): Name of the function found in the predict module that will be executed run predictions through model. It is not executed when this function is run.
            train_module_name (str): Full path of the module that contains the train function from the root of the zip.
            predict_module_name (str): Full path of the module that contains the predict function from the root of the zip.
            training_input_tables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            name (str): The name you want your model to have. Defaults to "<Project Name> Model".
            cpu_size (str): Size of the cpu for the model training function
            memory (int): Memory (in GB) for the model training function
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency

        Returns:
            Upload: None
        """
        return self.client.create_model_from_zip(self.project_id, train_function_name, predict_function_name, train_module_name, predict_module_name, training_input_tables, name, cpu_size, memory, package_requirements)

    def create_model_from_git(self, application_connector_id: str, branch_name: str, train_function_name: str, predict_function_name: str, train_module_name: str, predict_module_name: str, training_input_tables: list, python_root: str = None, name: str = None, cpu_size: str = None, memory: int = None, package_requirements: dict = None):
        """
        Initializes a new Model from a user provided git repository containing Python code. If a list of input feature groups are supplied,

        we will provide as arguments to the train and predict functions with the materialized feature groups for those
        input feature groups.

        This method expects `trainModuleName` and `predictModuleName` to be valid language source files which contains the functions named
        `trainFunctionName` and `predictFunctionName`, respectively. `trainFunctionName` returns the ModelVersion that is the result of
        training the model using `trainFunctionName` and `predictFunctionName` has no well defined return type,
        as it returns the prediction made by the `predictFunctionName`, which can be anything


        Args:
            application_connector_id (str): The unique ID associated with the git application connector.
            branch_name (str): Name of the branch in the git repository to be used for training.
            train_function_name (str): Name of the function found in train module that will be executed to train the model. It is not executed when this function is run.
            predict_function_name (str): Name of the function found in the predict module that will be executed run predictions through model. It is not executed when this function is run.
            train_module_name (str): Full path of the module that contains the train function from the root of the zip.
            predict_module_name (str): Full path of the module that contains the predict function from the root of the zip.
            training_input_tables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            python_root (str): Path from the top level of the git repository to the directory containing the Python source code. If not provided, the default is the root of the git repository.
            name (str): The name you want your model to have. Defaults to "<Project Name> Model".
            cpu_size (str): Size of the cpu for the model training function
            memory (int): Memory (in GB) for the model training function
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency

        Returns:
            Model: None
        """
        return self.client.create_model_from_git(self.project_id, application_connector_id, branch_name, train_function_name, predict_function_name, train_module_name, predict_module_name, training_input_tables, python_root, name, cpu_size, memory, package_requirements)

    def list_models(self):
        """
        Retrieves the list of models in the specified project.

        Args:
            project_id (str): The unique ID associated with the project.

        Returns:
            Model: An array of models.
        """
        return self.client.list_models(self.project_id)

    def create_model_monitor(self, training_feature_group_id: str, prediction_feature_group_id: str, name: str = None, refresh_schedule: str = None, target_value: str = None, feature_mappings: dict = None):
        """
        Runs a model monitor for the specified project.

        Args:
            training_feature_group_id (str): The unique ID of the training data feature group
            prediction_feature_group_id (str): The unique ID of the prediction data feature group
            name (str): The name you want your model monitor to have. Defaults to "<Project Name> Model Monitor".
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically retrain the created model monitor
            target_value (str): A target positive value for the label to compute bias for
            feature_mappings (dict): A json map to override features for prediction_feature_group, where keys are column names and the values are feature data use types.

        Returns:
            ModelMonitor: The new model monitor that was created.
        """
        return self.client.create_model_monitor(self.project_id, training_feature_group_id, prediction_feature_group_id, name, refresh_schedule, target_value, feature_mappings)

    def list_model_monitors(self):
        """
        Retrieves the list of models monitors in the specified project.

        Args:
            project_id (str): The unique ID associated with the project.

        Returns:
            ModelMonitor: An array of model monitors.
        """
        return self.client.list_model_monitors(self.project_id)

    def create_deployment_token(self):
        """
        Creates a deployment token for the specified project.

        Deployment tokens are used to authenticate requests to the prediction APIs and are scoped on the project level.


        Args:
            project_id (str): The unique ID associated with the project.

        Returns:
            DeploymentAuthToken: The deployment token.
        """
        return self.client.create_deployment_token(self.project_id)

    def list_deployments(self):
        """
        Retrieves a list of all deployments in the specified project.

        Args:
            project_id (str): The unique ID associated with the project.

        Returns:
            Deployment: An array of deployments.
        """
        return self.client.list_deployments(self.project_id)

    def list_deployment_tokens(self):
        """
        Retrieves a list of all deployment tokens in the specified project.

        Args:
            project_id (str): The unique ID associated with the project.

        Returns:
            DeploymentAuthToken: An array of deployment tokens.
        """
        return self.client.list_deployment_tokens(self.project_id)

    def list_refresh_policies(self, dataset_ids: list = [], model_ids: list = [], deployment_ids: list = [], batch_prediction_ids: list = [], model_monitor_ids: list = [], prediction_metric_ids: list = []):
        """
        List the refresh policies for the organization

        Args:
            dataset_ids (list): Comma separated list of Dataset IDs
            model_ids (list): Comma separated list of Model IDs
            deployment_ids (list): Comma separated list of Deployment IDs
            batch_prediction_ids (list): Comma separated list of Batch Prediction IDs
            model_monitor_ids (list): Comma separated list of Model Monitor IDs.
            prediction_metric_ids (list): Comma separated list of Prediction Metric IDs,

        Returns:
            RefreshPolicy: List of all refresh policies in the organization
        """
        return self.client.list_refresh_policies(self.project_id, dataset_ids, model_ids, deployment_ids, batch_prediction_ids, model_monitor_ids, prediction_metric_ids)

    def list_batch_predictions(self):
        """
        Retrieves a list for the batch predictions in the project

        Args:
            project_id (str): The unique identifier of the project

        Returns:
            BatchPrediction: A list of batch prediction jobs.
        """
        return self.client.list_batch_predictions(self.project_id)

    def attach_dataset(self, dataset_id, project_dataset_type):
        """
        Attaches dataset to the project.

        Args:
            dataset_id (unique string identifier): A unique identifier for the dataset.
            project_dataset_type (enum of type string): The unique use case specific dataset type that might be required or recommended for the specific use case.

        Returns:
            Schema: The schema of the attached dataset.
        """
        return self.client.attach_dataset_to_project(dataset_id, self.project_id, project_dataset_type)

    def remove_dataset(self, dataset_id):
        """
        Removes dataset from the project.

        Args:
            dataset_id (unique string identifier): A unique identifier for the dataset.
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
            Model: The model object.
        """
        return self.client.create_model_from_functions(self.project_id, train_function, predict_function, training_input_tables)
