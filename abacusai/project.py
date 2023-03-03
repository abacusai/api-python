from .return_class import AbstractApiClass


class Project(AbstractApiClass):
    """
        A project is a container which holds datasets, models and deployments

        Args:
            client (ApiClient): An authenticated API Client instance
            projectId (str): The ID of the project.
            name (str): The name of the project.
            useCase (str): The use case associated with the project.
            problemType (str): The problem type associated with the project.
            createdAt (str): The date and time when the project was created.
            featureGroupsEnabled (bool): Project uses feature groups instead of datasets.
    """

    def __init__(self, client, projectId=None, name=None, useCase=None, problemType=None, createdAt=None, featureGroupsEnabled=None):
        super().__init__(client, projectId)
        self.project_id = projectId
        self.name = name
        self.use_case = useCase
        self.problem_type = problemType
        self.created_at = createdAt
        self.feature_groups_enabled = featureGroupsEnabled

    def __repr__(self):
        return f"Project(project_id={repr(self.project_id)},\n  name={repr(self.name)},\n  use_case={repr(self.use_case)},\n  problem_type={repr(self.problem_type)},\n  created_at={repr(self.created_at)},\n  feature_groups_enabled={repr(self.feature_groups_enabled)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'project_id': self.project_id, 'name': self.name, 'use_case': self.use_case, 'problem_type': self.problem_type, 'created_at': self.created_at, 'feature_groups_enabled': self.feature_groups_enabled}

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
            project_id (str): A unique string identifier for the project.

        Returns:
            Project: The description of the project.
        """
        return self.client.describe_project(self.project_id)

    def list_datasets(self):
        """
        Retrieves all datasets associated with a specified project. This API returns all attributes of each dataset, such as its name, type, and ID.

        Args:
            project_id (str): Unique string identifier associated with the project.

        Returns:
            ProjectDataset: A list representing all of the datasets attached to the project.
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
        Delete a specified project from your organization.

        This method deletes the project, its associated trained models, and deployments. The datasets attached to the specified project remain available for use with other projects in the organization.

        This method will not delete a project that contains active deployments. Ensure that all active deployments are stopped before using the delete option.

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
            nested_column_name (str): The name of the nested column if the input feature is part of a nested feature group for the given feature_group_id.

        Returns:
            Feature: A list of objects that describes the resulting feature group's schema after the feature's featureMapping is set.
        """
        return self.client.set_feature_mapping(self.project_id, feature_group_id, feature_name, feature_mapping, nested_column_name)

    def validate(self, feature_group_ids: list = None):
        """
        Validates that the specified project has all required feature group types for its use case and that all required feature columns are set.

        Args:
            feature_group_ids (list): The list of feature group IDs to validate.

        Returns:
            ProjectValidation: The project validation. If the specified project is missing required columns or feature groups, the response includes an array of objects for each missing required feature group and the missing required features in each feature group.
        """
        return self.client.validate_project(self.project_id, feature_group_ids)

    def set_column_data_type(self, dataset_id: str, column: str, data_type: str):
        """
        Set the data type of a dataset's column.

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            column (str): The name of the column.
            data_type (str): The type of data in the column. Refer to the [guide on feature types](FEATURE_TYPES_URL) for more information. Note: Some ColumnMappings will restrict the options or explicitly set the DataType. Possible values:  CATEGORICAL,  CATEGORICAL_LIST,  NUMERICAL,  TIMESTAMP,  TEXT,  EMAIL,  LABEL_LIST,  JSON,  OBJECT_REFERENCE,  MULTICATEGORICAL_LIST,  COORDINATE_LIST,  NUMERICAL_LIST,  TIMESTAMP_LIST

        Returns:
            Schema: A list of objects that describes the resulting dataset's schema after the column's data type is set.
        """
        return self.client.set_column_data_type(self.project_id, dataset_id, column, data_type)

    def set_column_mapping(self, dataset_id: str, column: str, column_mapping: str):
        """
        Set a dataset's column mapping. If the column mapping is single-use and already set in another column in this dataset, this call will first remove the other column's mapping and move it to this column.

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            column (str): The name of the column.
            column_mapping (str): The mapping of the column in the dataset. See a list of column mapping enums here.

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
            filter_feature_group_use (str): The feature group use filter, when given as an argument, only allows feature groups present in this project to be returned if they are of the given use. Possible values are:  DATA_WRANGLING,  TRAINING_INPUT,  BATCH_PREDICTION_INPUT,  BATCH_PREDICTION_OUTPUT.

        Returns:
            FeatureGroup: All the Feature Groups in a project.
        """
        return self.client.list_project_feature_groups(self.project_id, filter_feature_group_use)

    def list_feature_group_templates(self, limit: int = 100, start_after_id: str = None, should_include_all_system_templates: bool = False):
        """
        List feature group templates for feature groups associated with the project.

        Args:
            limit (int): Maximum number of templates to be retrieved.
            start_after_id (str): Offset parameter to exclude all templates till the specified feature group template ID.
            should_include_all_system_templates (bool): If True, will include built-in templates.

        Returns:
            FeatureGroupTemplate: All the feature groups in the organization, optionally limited by the feature group that created the template(s).
        """
        return self.client.list_project_feature_group_templates(self.project_id, limit, start_after_id, should_include_all_system_templates)

    def get_training_config_options(self, feature_group_ids: list = None, for_retrain: bool = False, current_training_config: dict = None):
        """
        Retrieves the full initial description of the model training configuration options available for the specified project. The configuration options available are determined by the use case associated with the specified project. Refer to the [Use Case Documentation]({USE_CASES_URL}) for more information on use cases and use case-specific configuration options.

        Args:
            feature_group_ids (list): The feature group IDs to be used for training.
            for_retrain (bool): Whether the training config options are used for retraining.
            current_training_config (dict): The current state of the training config, with some options set, which shall be used to get new options after refresh. This is `None` by default initially.

        Returns:
            TrainingConfigOptions: An array of options that can be specified when training a model in this project.
        """
        return self.client.get_training_config_options(self.project_id, feature_group_ids, for_retrain, current_training_config)

    def create_train_test_data_split_feature_group(self, training_config: dict, feature_group_ids: list):
        """
        Get the train and test data split without training the model. Only supported for models with custom algorithms.

        Args:
            training_config (dict): The training config key/value pairs used to influence how the split is calculated.
            feature_group_ids (list): List of feature group IDs provided by the user, including the required one for data split and others to influence how to split.

        Returns:
            FeatureGroup: The feature group containing the training data and folds information.
        """
        return self.client.create_train_test_data_split_feature_group(self.project_id, training_config, feature_group_ids)

    def train_model(self, name: str = None, training_config: dict = None, feature_group_ids: list = None, refresh_schedule: str = None, custom_algorithms: list = None, custom_algorithms_only: bool = False, custom_algorithm_configs: dict = None, builtin_algorithms: list = None, cpu_size: str = None, memory: int = None):
        """
        Train a model for the specified project.

        This method trains a model in the given project, using user-specified training configurations defined in the `getTrainingConfigOptions` method.


        Args:
            name (str): The name of the model. Defaults to "<Project Name> Model".
            training_config (dict): The training config key/value pairs used to train this model.
            feature_group_ids (list): List of feature group IDs provided by the user to train the model on.
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically retrain the created model.
            custom_algorithms (list): List of user-defined algorithms to train. If not set, the default enabled custom algorithms will be used.
            custom_algorithms_only (bool): Whether to only run custom algorithms.
            custom_algorithm_configs (dict): Configs for each user-defined algorithm; key is the algorithm name, value is the config serialized to JSON.
            builtin_algorithms (list): List of IDs of the builtin algorithms provided by Abacus.AI to train. If not set, all applicable builtin algorithms will be used.
            cpu_size (str): Size of the CPU for the user-defined algorithms during training.
            memory (int): Memory (in GB) for the user-defined algorithms during training.

        Returns:
            Model: The new model which is being trained.
        """
        return self.client.train_model(self.project_id, name, training_config, feature_group_ids, refresh_schedule, custom_algorithms, custom_algorithms_only, custom_algorithm_configs, builtin_algorithms, cpu_size, memory)

    def create_model_from_python(self, function_source_code: str, train_function_name: str, training_input_tables: list, predict_function_name: str = None, predict_many_function_name: str = None, initialize_function_name: str = None, name: str = None, cpu_size: str = None, memory: int = None, training_config: dict = None, exclusive_run: bool = False, package_requirements: list = None, use_gpu: bool = False):
        """
        Initializes a new Model from user-provided Python code. If a list of input feature groups is supplied, they will be provided as arguments to the train and predict functions with the materialized feature groups for those input feature groups.

        This method expects `functionSourceCode` to be a valid language source file which contains the functions named `trainFunctionName` and `predictFunctionName`. `trainFunctionName` returns the ModelVersion that is the result of training the model using `trainFunctionName` and `predictFunctionName` has no well-defined return type, as it returns the prediction made by the `predictFunctionName`, which can be anything.


        Args:
            function_source_code (str): Contents of a valid Python source code file. The source code should contain the functions named `trainFunctionName` and `predictFunctionName`. A list of allowed import and system libraries for each language is specified in the user functions documentation section.
            train_function_name (str): Name of the function found in the source code that will be executed to train the model. It is not executed when this function is run.
            training_input_tables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            predict_function_name (str): Name of the function found in the source code that will be executed to run predictions through the model. It is not executed when this function is run.
            predict_many_function_name (str): Name of the function found in the source code that will be executed for batch prediction of the model. It is not executed when this function is run.
            initialize_function_name (str): Name of the function found in the source code to initialize the trained model before using it to make predictions using the model
            name (str): The name you want your model to have. Defaults to "<Project Name> Model"
            cpu_size (str): Size of the CPU for the model training function
            memory (int): Memory (in GB) for the model training function
            training_config (dict): Training configuration
            exclusive_run (bool): Decides if this model will be run exclusively or along with other Abacus.ai algorithms
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            use_gpu (bool): Whether this model needs gpu

        Returns:
            Model: The new model, which has not been trained.
        """
        return self.client.create_model_from_python(self.project_id, function_source_code, train_function_name, training_input_tables, predict_function_name, predict_many_function_name, initialize_function_name, name, cpu_size, memory, training_config, exclusive_run, package_requirements, use_gpu)

    def list_models(self):
        """
        Retrieves the list of models in the specified project.

        Args:
            project_id (str): Unique string identifier associated with the project.

        Returns:
            Model: A list of models.
        """
        return self.client.list_models(self.project_id)

    def get_custom_train_function_info(self, feature_group_names_for_training: list = None, training_data_parameter_name_override: dict = None, training_config: dict = None, custom_algorithm_config: dict = None):
        """
        Returns information about how to call the custom train function.

        Args:
            feature_group_names_for_training (list): A list of feature group table names to be used for training.
            training_data_parameter_name_override (dict): Override from feature group type to parameter name in the train function.
            training_config (dict): Training config names and values for the options supported by the Abacus.ai platform.
            custom_algorithm_config (dict): User-defined config that can be serialized by JSON.

        Returns:
            CustomTrainFunctionInfo: Information about how to call the customer-provided train function.
        """
        return self.client.get_custom_train_function_info(self.project_id, feature_group_names_for_training, training_data_parameter_name_override, training_config, custom_algorithm_config)

    def create_model_monitor(self, prediction_feature_group_id: str, training_feature_group_id: str = None, name: str = None, refresh_schedule: str = None, target_value: str = None, target_value_bias: str = None, target_value_performance: str = None, feature_mappings: dict = None, model_id: str = None, training_feature_mappings: dict = None, feature_group_base_monitor_config: dict = None, feature_group_comparison_monitor_config: dict = None):
        """
        Runs a model monitor for the specified project.

        Args:
            prediction_feature_group_id (str): The unique ID of the prediction data feature group.
            training_feature_group_id (str): The unique ID of the training data feature group.
            name (str): The name you want your model monitor to have. Defaults to "<Project Name> Model Monitor".
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically retrain the created model monitor.
            target_value (str): A target positive value for the label to compute bias and PR/AUC for performance page (old style until UI is on prod).
            target_value_bias (str): A target positive value for the label to compute bias.
            target_value_performance (str): A target positive value for the label to compute PR curve/AUC for performance page.
            feature_mappings (dict): A JSON map to override features for prediction_feature_group, where keys are column names and the values are feature data use types.
            model_id (str): The unique ID of the model.
            training_feature_mappings (dict): A JSON map to override features for training_fature_group, where keys are column names and the values are feature data use types.
            feature_group_base_monitor_config (dict): Selection strategy for the feature_group 1 with the feature group version if selected.
            feature_group_comparison_monitor_config (dict): Selection strategy for the feature_group 1 with the feature group version if selected.

        Returns:
            ModelMonitor: The new model monitor that was created.
        """
        return self.client.create_model_monitor(self.project_id, prediction_feature_group_id, training_feature_group_id, name, refresh_schedule, target_value, target_value_bias, target_value_performance, feature_mappings, model_id, training_feature_mappings, feature_group_base_monitor_config, feature_group_comparison_monitor_config)

    def list_model_monitors(self):
        """
        Retrieves the list of model monitors in the specified project.

        Args:
            project_id (str): Unique string identifier associated with the project.

        Returns:
            ModelMonitor: A list of model monitors.
        """
        return self.client.list_model_monitors(self.project_id)

    def create_vision_drift_monitor(self, prediction_feature_group_id: str, training_feature_group_id: str, name: str, feature_mappings: dict, training_feature_mappings: dict, target_value_performance: str = None, refresh_schedule: str = None):
        """
        Runs a vision drift monitor for the specified project.

        Args:
            prediction_feature_group_id (str): Unique string identifier of the prediction data feature group.
            training_feature_group_id (str): Unique string identifier of the training data feature group.
            name (str): The name you want your model monitor to have. Defaults to "<Project Name> Model Monitor".
            feature_mappings (dict): A JSON map to override features for prediction_feature_group, where keys are column names and the values are feature data use types.
            training_feature_mappings (dict): A JSON map to override features for training_feature_group, where keys are column names and the values are feature data use types.
            target_value_performance (str): A target positive value for the label to compute precision-recall curve/area under curve for performance page.
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically retrain the created vision drift monitor.

        Returns:
            ModelMonitor: The new model monitor that was created.
        """
        return self.client.create_vision_drift_monitor(self.project_id, prediction_feature_group_id, training_feature_group_id, name, feature_mappings, training_feature_mappings, target_value_performance, refresh_schedule)

    def create_eda(self, feature_group_id: str, name: str, refresh_schedule: str = None, include_collinearity: bool = False, include_data_consistency: bool = False, collinearity_keys: list = None, primary_keys: list = None, data_consistency_test_config: dict = None, data_consistency_reference_config: dict = None, feature_mappings: dict = None, forecast_frequency: str = None):
        """
        Run an Exploratory Data Analysis (EDA) for the specified project.

        Args:
            feature_group_id (str): The unique ID of the prediction data feature group.
            name (str): The name you want your model monitor to have. Defaults to "<Project Name> EDA".
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically retrain the created EDA.
            include_collinearity (bool): Set to True if the EDA type is collinearity.
            include_data_consistency (bool): Set to True if the EDA type is data consistency.
            collinearity_keys (list): List of features to use for collinearity
            primary_keys (list): List of features that corresponds to the primary keys or item ids for the given feature group for Data Consistency analysis or Forecasting analysis respectively.
            data_consistency_test_config (dict): Test feature group version selection strategy for Data Consistency EDA type.
            data_consistency_reference_config (dict): Reference feature group version selection strategy for Data Consistency EDA type.
            feature_mappings (dict): A JSON map to override features for the given feature_group, where keys are column names and the values are feature data use types. (In forecasting, used to set the timestamp column and target value)
            forecast_frequency (str): The frequency of the data. It can be either HOURLY, DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY.

        Returns:
            Eda: The new EDA object that was created.
        """
        return self.client.create_eda(self.project_id, feature_group_id, name, refresh_schedule, include_collinearity, include_data_consistency, collinearity_keys, primary_keys, data_consistency_test_config, data_consistency_reference_config, feature_mappings, forecast_frequency)

    def list_eda(self):
        """
        Retrieves the list of Exploratory Data Analysis (EDA) in the specified project.

        Args:
            project_id (str): Unique string identifier associated with the project.

        Returns:
            Eda: List of EDA objects.
        """
        return self.client.list_eda(self.project_id)

    def create_monitor_alert(self, model_monitor_id: str, alert_name: str, condition_config: dict, action_config: dict):
        """
        Create a monitor alert for the given conditions and monitor

        Args:
            model_monitor_id (str): Unique string identifier for the model monitor created under the project.
            alert_name (str): Name of the alert.
            condition_config (dict): Condition to run the actions for the alert.
            action_config (dict): Configuration for the action of the alert.

        Returns:
            MonitorAlert: Object describing the monitor alert.
        """
        return self.client.create_monitor_alert(self.project_id, model_monitor_id, alert_name, condition_config, action_config)

    def create_deployment_token(self, name: str = None):
        """
        Creates a deployment token for the specified project.

        Deployment tokens are used to authenticate requests to the prediction APIs and are scoped to the project level.


        Args:
            name (str): The name of the deployment token.

        Returns:
            DeploymentAuthToken: The deployment token.
        """
        return self.client.create_deployment_token(self.project_id, name)

    def list_deployments(self):
        """
        Retrieves a list of all deployments in the specified project.

        Args:
            project_id (str): The unique identifier associated with the project.

        Returns:
            Deployment: An array of deployments.
        """
        return self.client.list_deployments(self.project_id)

    def list_deployment_tokens(self):
        """
        Retrieves a list of all deployment tokens associated with the specified project.

        Args:
            project_id (str): The unique ID associated with the project.

        Returns:
            DeploymentAuthToken: A list of deployment tokens.
        """
        return self.client.list_deployment_tokens(self.project_id)

    def list_refresh_policies(self, dataset_ids: list = [], model_ids: list = [], deployment_ids: list = [], batch_prediction_ids: list = [], model_monitor_ids: list = [], prediction_metric_ids: list = []):
        """
        List the refresh policies for the organization

        Args:
            dataset_ids (list): Comma-separated list of Dataset IDs.
            model_ids (list): Comma-separated list of Model IDs.
            deployment_ids (list): Comma-separated list of Deployment IDs.
            batch_prediction_ids (list): Comma-separated list of Batch Prediction IDs.
            model_monitor_ids (list): Comma-separated list of Model Monitor IDs.
            prediction_metric_ids (list): Comma-separated list of Prediction Metric IDs.

        Returns:
            RefreshPolicy: List of all refresh policies in the organization.
        """
        return self.client.list_refresh_policies(self.project_id, dataset_ids, model_ids, deployment_ids, batch_prediction_ids, model_monitor_ids, prediction_metric_ids)

    def list_batch_predictions(self):
        """
        Retrieves a list of batch predictions in the project.

        Args:
            project_id (str): Unique string identifier of the project.

        Returns:
            BatchPrediction: List of batch prediction jobs.
        """
        return self.client.list_batch_predictions(self.project_id)

    def create_graph_dashboard(self, name: str, python_function_ids: list = None):
        """
        Create a plot dashboard given selected python plots

        Args:
            name (str): The name of the dashboard.
            python_function_ids (list): A list of unique string identifiers for the python functions to be used in the graph dashboard.

        Returns:
            GraphDashboard: An object describing the graph dashboard.
        """
        return self.client.create_graph_dashboard(self.project_id, name, python_function_ids)

    def list_graph_dashboards(self):
        """
        Lists the graph dashboards for a project

        Args:
            project_id (str): Unique string identifier for the project to list graph dashboards from.

        Returns:
            GraphDashboard: A list of graph dashboards.
        """
        return self.client.list_graph_dashboards(self.project_id)

    def list_builtin_algorithms(self, feature_group_ids: list = None, training_config: dict = None):
        """
        Return list of built-in algorithms based on given input.

        Args:
            feature_group_ids (list): List of feature group identifiers applied to the algorithms.
            training_config (dict): The training configuration key/value pairs used to train with the algorithm.

        Returns:
            Algorithm: List of applicable builtin algorithms.
        """
        return self.client.list_builtin_algorithms(self.project_id, feature_group_ids, training_config)

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

    def create_model_from_functions(self, train_function: callable, predict_function: callable = None, training_input_tables: list = None, predict_many_function: callable = None, initialize_function: callable = None, cpu_size: str = None, memory: int = None, training_config: dict = None, exclusive_run: bool = False):
        """
        Creates a model using python.

        Args:
            train_function (callable): The train function is passed.
            predict_function (callable): The prediction function is passed.
            training_input_tables (list, optional): The input tables to be used for training the model. Defaults to None.
            predict_many_function (callable): Prediction function for batch input
            cpu_size (str): Size of the cpu for the feature group function
            memory (int): Memory (in GB) for the feature group function

        Returns:
            Model: The model object.
        """
        return self.client.create_model_from_functions(project_id=self.id, train_function=train_function, predict_function=predict_function, training_input_tables=training_input_tables, predict_many_function=predict_many_function, initialize_function=initialize_function, training_config=training_config, cpu_size=cpu_size, memory=memory, exclusive_run=exclusive_run)
