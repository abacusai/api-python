from typing import Dict, List, Union

from .api_class import (
    AgentInterface, AlertActionConfig, AlertConditionConfig,
    ForecastingMonitorConfig, TrainingConfig, VectorStoreConfig, WorkflowGraph
)
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
            tags (list[str]): List of tags associated with the project.
    """

    def __init__(self, client, projectId=None, name=None, useCase=None, problemType=None, createdAt=None, tags=None):
        super().__init__(client, projectId)
        self.project_id = projectId
        self.name = name
        self.use_case = useCase
        self.problem_type = problemType
        self.created_at = createdAt
        self.tags = tags
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'project_id': repr(self.project_id), f'name': repr(self.name), f'use_case': repr(
            self.use_case), f'problem_type': repr(self.problem_type), f'created_at': repr(self.created_at), f'tags': repr(self.tags)}
        class_name = "Project"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'project_id': self.project_id, 'name': self.name, 'use_case': self.use_case,
                'problem_type': self.problem_type, 'created_at': self.created_at, 'tags': self.tags}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

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

    def rename(self, name: str):
        """
        This method renames a project after it is created.

        Args:
            name (str): The new name for the project.
        """
        return self.client.rename_project(self.project_id, name)

    def delete(self, force_delete: bool = False):
        """
        Delete a specified project from your organization.

        This method deletes the project, its associated trained models, and deployments. The datasets attached to the specified project remain available for use with other projects in the organization.

        This method will not delete a project that contains active deployments. Ensure that all active deployments are stopped before using the delete option.

        Note: All projects, models, and deployments cannot be recovered once they are deleted.


        Args:
            force_delete (bool): If True, the project will be deleted even if it has active deployments.
        """
        return self.client.delete_project(self.project_id, force_delete)

    def add_tags(self, tags: list):
        """
        This method adds a tag to a project.

        Args:
            tags (list): The tags to add to the project.
        """
        return self.client.add_project_tags(self.project_id, tags)

    def remove_tags(self, tags: list):
        """
        This method removes a tag from a project.

        Args:
            tags (list): The tags to remove from the project.
        """
        return self.client.remove_project_tags(self.project_id, tags)

    def set_feature_mapping(self, feature_group_id: str, feature_name: str, feature_mapping: str = None, nested_column_name: str = None):
        """
        Set a column's feature mapping. If the column mapping is single-use and already set in another column in this feature group, this call will first remove the other column's mapping and move it to this column.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            feature_name (str): The name of the feature.
            feature_mapping (str): The mapping of the feature in the feature group.
            nested_column_name (str): The name of the nested column if the input feature is part of a nested feature group for the given feature_group_id.

        Returns:
            list[Feature]: A list of objects that describes the resulting feature group's schema after the feature's featureMapping is set.
        """
        return self.client.set_feature_mapping(self.project_id, feature_group_id, feature_name, feature_mapping, nested_column_name)

    def validate(self, feature_group_ids: List = None):
        """
        Validates that the specified project has all required feature group types for its use case and that all required feature columns are set.

        Args:
            feature_group_ids (List): The list of feature group IDs to validate.

        Returns:
            ProjectValidation: The project validation. If the specified project is missing required columns or feature groups, the response includes an array of objects for each missing required feature group and the missing required features in each feature group.
        """
        return self.client.validate_project(self.project_id, feature_group_ids)

    def infer_feature_mappings(self, feature_group_id: str):
        """
        Infer the feature mappings for the feature group in the project based on the problem type.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.

        Returns:
            InferredFeatureMappings: A dict that contains the inferred feature mappings.
        """
        return self.client.infer_feature_mappings(self.project_id, feature_group_id)

    def add_scope_for_user(self, email: str, scope: list):
        """
        Add a user to a project.

        Args:
            email (str): The user's email.
            scope (list): The list of project scopes.
        """
        return self.client.add_project_scope_for_user(self.project_id, email, scope)

    def describe_feature_group(self, feature_group_id: str):
        """
        Describe a feature group associated with a project

        Args:
            feature_group_id (str): The unique ID associated with the feature group.

        Returns:
            ProjectFeatureGroup: The project feature group object.
        """
        return self.client.describe_project_feature_group(self.project_id, feature_group_id)

    def list_feature_groups(self, filter_feature_group_use: str = None, limit: int = 100, start_after_id: str = None):
        """
        List all the feature groups associated with a project

        Args:
            filter_feature_group_use (str): The feature group use filter, when given as an argument only allows feature groups present in this project to be returned if they are of the given use. Possible values are: 'USER_CREATED', 'BATCH_PREDICTION_OUTPUT'.
            limit (int): The maximum number of feature groups to be retrieved.
            start_after_id (str): An offset parameter to exclude all feature groups up to a specified ID.

        Returns:
            list[ProjectFeatureGroup]: All the Feature Groups in a project.
        """
        return self.client.list_project_feature_groups(self.project_id, filter_feature_group_use, limit, start_after_id)

    def list_feature_group_templates(self, limit: int = 100, start_after_id: str = None, should_include_all_system_templates: bool = False):
        """
        List feature group templates for feature groups associated with the project.

        Args:
            limit (int): Maximum number of templates to be retrieved.
            start_after_id (str): Offset parameter to exclude all templates till the specified feature group template ID.
            should_include_all_system_templates (bool): If True, will include built-in templates.

        Returns:
            list[FeatureGroupTemplate]: All the feature groups in the organization, optionally limited by the feature group that created the template(s).
        """
        return self.client.list_project_feature_group_templates(self.project_id, limit, start_after_id, should_include_all_system_templates)

    def get_training_config_options(self, feature_group_ids: List = None, for_retrain: bool = False, current_training_config: Union[dict, TrainingConfig] = None):
        """
        Retrieves the full initial description of the model training configuration options available for the specified project. The configuration options available are determined by the use case associated with the specified project. Refer to the [Use Case Documentation]({USE_CASES_URL}) for more information on use cases and use case-specific configuration options.

        Args:
            feature_group_ids (List): The feature group IDs to be used for training.
            for_retrain (bool): Whether the training config options are used for retraining.
            current_training_config (TrainingConfig): The current state of the training config, with some options set, which shall be used to get new options after refresh. This is `None` by default initially.

        Returns:
            list[TrainingConfigOptions]: An array of options that can be specified when training a model in this project.
        """
        return self.client.get_training_config_options(self.project_id, feature_group_ids, for_retrain, current_training_config)

    def create_train_test_data_split_feature_group(self, training_config: Union[dict, TrainingConfig], feature_group_ids: List):
        """
        Get the train and test data split without training the model. Only supported for models with custom algorithms.

        Args:
            training_config (TrainingConfig): The training config used to influence how the split is calculated.
            feature_group_ids (List): List of feature group IDs provided by the user, including the required one for data split and others to influence how to split.

        Returns:
            FeatureGroup: The feature group containing the training data and folds information.
        """
        return self.client.create_train_test_data_split_feature_group(self.project_id, training_config, feature_group_ids)

    def train_model(self, name: str = None, training_config: Union[dict, TrainingConfig] = None, feature_group_ids: List = None, refresh_schedule: str = None, custom_algorithms: list = None, custom_algorithms_only: bool = False, custom_algorithm_configs: dict = None, builtin_algorithms: list = None, cpu_size: str = None, memory: int = None, algorithm_training_configs: list = None):
        """
        Create a new model and start its training in the given project.

        Args:
            name (str): The name of the model. Defaults to "<Project Name> Model".
            training_config (TrainingConfig): The training config used to train this model.
            feature_group_ids (List): List of feature group IDs provided by the user to train the model on.
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically retrain the created model.
            custom_algorithms (list): List of user-defined algorithms to train. If not set, the default enabled custom algorithms will be used.
            custom_algorithms_only (bool): Whether to only run custom algorithms.
            custom_algorithm_configs (dict): Configs for each user-defined algorithm; key is the algorithm name, value is the config serialized to JSON.
            builtin_algorithms (list): List of algorithm names or algorithm IDs of the builtin algorithms provided by Abacus.AI to train. If not set, all applicable builtin algorithms will be used.
            cpu_size (str): Size of the CPU for the user-defined algorithms during training.
            memory (int): Memory (in GB) for the user-defined algorithms during training.
            algorithm_training_configs (list): List of algorithm specifc training configs that will be part of the model training AutoML run.

        Returns:
            Model: The new model which is being trained.
        """
        return self.client.train_model(self.project_id, name, training_config, feature_group_ids, refresh_schedule, custom_algorithms, custom_algorithms_only, custom_algorithm_configs, builtin_algorithms, cpu_size, memory, algorithm_training_configs)

    def create_model_from_python(self, function_source_code: str, train_function_name: str, training_input_tables: list, predict_function_name: str = None, predict_many_function_name: str = None, initialize_function_name: str = None, name: str = None, cpu_size: str = None, memory: int = None, training_config: Union[dict, TrainingConfig] = None, exclusive_run: bool = False, package_requirements: list = None, use_gpu: bool = False, is_thread_safe: bool = None):
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
            training_config (TrainingConfig): Training configuration
            exclusive_run (bool): Decides if this model will be run exclusively or along with other Abacus.AI algorithms
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            use_gpu (bool): Whether this model needs gpu
            is_thread_safe (bool): Whether this model is thread safe

        Returns:
            Model: The new model, which has not been trained.
        """
        return self.client.create_model_from_python(self.project_id, function_source_code, train_function_name, training_input_tables, predict_function_name, predict_many_function_name, initialize_function_name, name, cpu_size, memory, training_config, exclusive_run, package_requirements, use_gpu, is_thread_safe)

    def list_models(self):
        """
        Retrieves the list of models in the specified project.

        Args:
            project_id (str): Unique string identifier associated with the project.

        Returns:
            list[Model]: A list of models.
        """
        return self.client.list_models(self.project_id)

    def get_custom_train_function_info(self, feature_group_names_for_training: list = None, training_data_parameter_name_override: dict = None, training_config: Union[dict, TrainingConfig] = None, custom_algorithm_config: dict = None):
        """
        Returns information about how to call the custom train function.

        Args:
            feature_group_names_for_training (list): A list of feature group table names to be used for training.
            training_data_parameter_name_override (dict): Override from feature group type to parameter name in the train function.
            training_config (TrainingConfig): Training config for the options supported by the Abacus.AI platform.
            custom_algorithm_config (dict): User-defined config that can be serialized by JSON.

        Returns:
            CustomTrainFunctionInfo: Information about how to call the customer-provided train function.
        """
        return self.client.get_custom_train_function_info(self.project_id, feature_group_names_for_training, training_data_parameter_name_override, training_config, custom_algorithm_config)

    def create_model_monitor(self, prediction_feature_group_id: str, training_feature_group_id: str = None, name: str = None, refresh_schedule: str = None, target_value: str = None, target_value_bias: str = None, target_value_performance: str = None, feature_mappings: dict = None, model_id: str = None, training_feature_mappings: dict = None, feature_group_base_monitor_config: dict = None, feature_group_comparison_monitor_config: dict = None, exclude_interactive_performance_analysis: bool = True, exclude_bias_analysis: bool = None, exclude_performance_analysis: bool = None, exclude_feature_drift_analysis: bool = None, exclude_data_integrity_analysis: bool = None):
        """
        Runs a model monitor for the specified project.

        Args:
            prediction_feature_group_id (str): The unique ID of the prediction data feature group.
            training_feature_group_id (str): The unique ID of the training data feature group.
            name (str): The name you want your model monitor to have. Defaults to "<Project Name> Model Monitor".
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically retrain the created model monitor.
            target_value (str): A target positive value for the label to compute bias and PR/AUC for performance page.
            target_value_bias (str): A target positive value for the label to compute bias.
            target_value_performance (str): A target positive value for the label to compute PR curve/AUC for performance page.
            feature_mappings (dict): A JSON map to override features for prediction_feature_group, where keys are column names and the values are feature data use types.
            model_id (str): The unique ID of the model.
            training_feature_mappings (dict): A JSON map to override features for training_fature_group, where keys are column names and the values are feature data use types.
            feature_group_base_monitor_config (dict): Selection strategy for the feature_group 1 with the feature group version if selected.
            feature_group_comparison_monitor_config (dict): Selection strategy for the feature_group 1 with the feature group version if selected.
            exclude_interactive_performance_analysis (bool): Whether to exclude interactive performance analysis. Defaults to True if not provided.
            exclude_bias_analysis (bool): Whether to exclude bias analysis in the model monitor. For default value bias analysis is included.
            exclude_performance_analysis (bool): Whether to exclude performance analysis in the model monitor. For default value performance analysis is included.
            exclude_feature_drift_analysis (bool): Whether to exclude feature drift analysis in the model monitor. For default value feature drift analysis is included.
            exclude_data_integrity_analysis (bool): Whether to exclude data integrity analysis in the model monitor. For default value data integrity analysis is included.

        Returns:
            ModelMonitor: The new model monitor that was created.
        """
        return self.client.create_model_monitor(self.project_id, prediction_feature_group_id, training_feature_group_id, name, refresh_schedule, target_value, target_value_bias, target_value_performance, feature_mappings, model_id, training_feature_mappings, feature_group_base_monitor_config, feature_group_comparison_monitor_config, exclude_interactive_performance_analysis, exclude_bias_analysis, exclude_performance_analysis, exclude_feature_drift_analysis, exclude_data_integrity_analysis)

    def list_model_monitors(self, limit: int = None):
        """
        Retrieves the list of model monitors in the specified project.

        Args:
            limit (int): Maximum number of model monitors to return. We'll have internal limit if not set.

        Returns:
            list[ModelMonitor]: A list of model monitors.
        """
        return self.client.list_model_monitors(self.project_id, limit)

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
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically rerun the created vision drift monitor.

        Returns:
            ModelMonitor: The new model monitor that was created.
        """
        return self.client.create_vision_drift_monitor(self.project_id, prediction_feature_group_id, training_feature_group_id, name, feature_mappings, training_feature_mappings, target_value_performance, refresh_schedule)

    def create_nlp_drift_monitor(self, prediction_feature_group_id: str, training_feature_group_id: str, name: str, feature_mappings: dict, training_feature_mappings: dict, target_value_performance: str = None, refresh_schedule: str = None):
        """
        Runs an NLP drift monitor for the specified project.

        Args:
            prediction_feature_group_id (str): Unique string identifier of the prediction data feature group.
            training_feature_group_id (str): Unique string identifier of the training data feature group.
            name (str): The name you want your model monitor to have. Defaults to "<Project Name> Model Monitor".
            feature_mappings (dict): A JSON map to override features for prediction_feature_group, where keys are column names and the values are feature data use types.
            training_feature_mappings (dict): A JSON map to override features for training_feature_group, where keys are column names and the values are feature data use types.
            target_value_performance (str): A target positive value for the label to compute precision-recall curve/area under curve for performance page.
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically rerun the created nlp drift monitor.

        Returns:
            ModelMonitor: The new model monitor that was created.
        """
        return self.client.create_nlp_drift_monitor(self.project_id, prediction_feature_group_id, training_feature_group_id, name, feature_mappings, training_feature_mappings, target_value_performance, refresh_schedule)

    def create_forecasting_monitor(self, name: str, prediction_feature_group_id: str, training_feature_group_id: str, training_forecast_config: Union[dict, ForecastingMonitorConfig], prediction_forecast_config: Union[dict, ForecastingMonitorConfig], forecast_frequency: str, refresh_schedule: str = None):
        """
        Runs a forecasting monitor for the specified project.

        Args:
            name (str): The name you want your model monitor to have. Defaults to "<Project Name> Model Monitor".
            prediction_feature_group_id (str): Unique string identifier of the prediction data feature group.
            training_feature_group_id (str): Unique string identifier of the training data feature group.
            training_forecast_config (ForecastingMonitorConfig): The configuration for the training data.
            prediction_forecast_config (ForecastingMonitorConfig): The configuration for the prediction data.
            forecast_frequency (str): The frequency of the forecast. Defaults to the frequency of the prediction data.
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically rerun the created forecasting monitor.

        Returns:
            ModelMonitor: The new model monitor that was created.
        """
        return self.client.create_forecasting_monitor(self.project_id, name, prediction_feature_group_id, training_feature_group_id, training_forecast_config, prediction_forecast_config, forecast_frequency, refresh_schedule)

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
            list[Eda]: List of EDA objects.
        """
        return self.client.list_eda(self.project_id)

    def list_holdout_analysis(self, model_id: str = None):
        """
        List holdout analyses for a project. Optionally, filter by model.

        Args:
            model_id (str): (optional) ID of the model to filter by

        Returns:
            list[HoldoutAnalysis]: The holdout analyses
        """
        return self.client.list_holdout_analysis(self.project_id, model_id)

    def create_monitor_alert(self, alert_name: str, condition_config: Union[dict, AlertConditionConfig], action_config: Union[dict, AlertActionConfig], model_monitor_id: str = None, realtime_monitor_id: str = None):
        """
        Create a monitor alert for the given conditions and monitor. We can create monitor alert either for model monitor or real-time monitor.

        Args:
            alert_name (str): Name of the alert.
            condition_config (AlertConditionConfig): Condition to run the actions for the alert.
            action_config (AlertActionConfig): Configuration for the action of the alert.
            model_monitor_id (str): Unique string identifier for the model monitor created under the project.
            realtime_monitor_id (str): Unique string identifier for the real-time monitor for the deployment created under the project.

        Returns:
            MonitorAlert: Object describing the monitor alert.
        """
        return self.client.create_monitor_alert(self.project_id, alert_name, condition_config, action_config, model_monitor_id, realtime_monitor_id)

    def list_prediction_operators(self):
        """
        List all the prediction operators inside a project.

        Args:
            project_id (str): The unique ID of the project.

        Returns:
            list[PredictionOperator]: A list of prediction operator objects.
        """
        return self.client.list_prediction_operators(self.project_id)

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
            list[Deployment]: An array of deployments.
        """
        return self.client.list_deployments(self.project_id)

    def list_deployment_tokens(self):
        """
        Retrieves a list of all deployment tokens associated with the specified project.

        Args:
            project_id (str): The unique ID associated with the project.

        Returns:
            list[DeploymentAuthToken]: A list of deployment tokens.
        """
        return self.client.list_deployment_tokens(self.project_id)

    def list_realtime_monitors(self):
        """
        List the real-time monitors associated with the deployment id.

        Args:
            project_id (str): Unique string identifier for the deployment.

        Returns:
            list[RealtimeMonitor]: An array of real-time monitors.
        """
        return self.client.list_realtime_monitors(self.project_id)

    def list_refresh_policies(self, dataset_ids: List = [], feature_group_id: str = None, model_ids: List = [], deployment_ids: List = [], batch_prediction_ids: List = [], model_monitor_ids: List = [], notebook_ids: List = []):
        """
        List the refresh policies for the organization. If no filters are specified, all refresh policies are returned.

        Args:
            dataset_ids (List): Comma-separated list of Dataset IDs.
            feature_group_id (str): Feature Group ID for which we wish to see the refresh policies attached.
            model_ids (List): Comma-separated list of Model IDs.
            deployment_ids (List): Comma-separated list of Deployment IDs.
            batch_prediction_ids (List): Comma-separated list of Batch Prediction IDs.
            model_monitor_ids (List): Comma-separated list of Model Monitor IDs.
            notebook_ids (List): Comma-separated list of Notebook IDs.

        Returns:
            list[RefreshPolicy]: List of all refresh policies in the organization.
        """
        return self.client.list_refresh_policies(self.project_id, dataset_ids, feature_group_id, model_ids, deployment_ids, batch_prediction_ids, model_monitor_ids, notebook_ids)

    def list_batch_predictions(self, limit: int = None):
        """
        Retrieves a list of batch predictions in the project.

        Args:
            limit (int): Maximum number of batch predictions to return. We'll have internal limit if not set.

        Returns:
            list[BatchPrediction]: List of batch prediction jobs.
        """
        return self.client.list_batch_predictions(self.project_id, limit)

    def list_pipelines(self):
        """
        Lists the pipelines for an organization or a project

        Args:
            project_id (str): Unique string identifier for the project to list graph dashboards from.

        Returns:
            list[Pipeline]: A list of pipelines.
        """
        return self.client.list_pipelines(self.project_id)

    def create_graph_dashboard(self, name: str, python_function_ids: List = None):
        """
        Create a plot dashboard given selected python plots

        Args:
            name (str): The name of the dashboard.
            python_function_ids (List): A list of unique string identifiers for the python functions to be used in the graph dashboard.

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
            list[GraphDashboard]: A list of graph dashboards.
        """
        return self.client.list_graph_dashboards(self.project_id)

    def list_builtin_algorithms(self, feature_group_ids: List, training_config: Union[dict, TrainingConfig] = None):
        """
        Return list of built-in algorithms based on given input data and training config.

        Args:
            feature_group_ids (List): List of feature group IDs specifying input data.
            training_config (TrainingConfig): The training config to be used for model training.

        Returns:
            list[Algorithm]: List of applicable builtin algorithms.
        """
        return self.client.list_builtin_algorithms(self.project_id, feature_group_ids, training_config)

    def create_chat_session(self, name: str = None):
        """
        Creates a chat session with Data Science Co-pilot.

        Args:
            name (str): The name of the chat session. Defaults to the project name.

        Returns:
            ChatSession: The chat session with Data Science Co-pilot
        """
        return self.client.create_chat_session(self.project_id, name)

    def create_agent(self, function_source_code: str = None, agent_function_name: str = None, name: str = None, memory: int = None, package_requirements: list = [], description: str = None, enable_binary_input: bool = False, evaluation_feature_group_id: str = None, agent_input_schema: dict = None, agent_output_schema: dict = None, workflow_graph: Union[dict, WorkflowGraph] = None, agent_interface: Union[dict, AgentInterface] = AgentInterface.DEFAULT, included_modules: List = None, org_level_connectors: List = None, user_level_connectors: Dict = None, initialize_function_name: str = None, initialize_function_code: str = None):
        """
        Creates a new AI agent using the given agent workflow graph definition.

        Args:
            name (str): The name you want your agent to have, defaults to "<Project Name> Agent".
            memory (int): Overrides the default memory allocation (in GB) for the agent.
            package_requirements (list): A list of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            description (str): A description of the agent, including its purpose and instructions.
            evaluation_feature_group_id (str): The ID of the feature group to use for evaluation.
            workflow_graph (WorkflowGraph): The workflow graph for the agent.
            agent_interface (AgentInterface): The interface that the agent will be deployed with.
            included_modules (List): A list of user created custom modules to include in the agent's environment.
            org_level_connectors (List): A list of org level connector ids to be used by the agent.
            user_level_connectors (Dict): A dictionary mapping ApplicationConnectorType keys to lists of OAuth scopes. Each key represents a specific user level application connector, while the value is a list of scopes that define the permissions granted to the application.
            initialize_function_name (str): The name of the function to be used for initialization.
            initialize_function_code (str): The function code to be used for initialization.

        Returns:
            Agent: The new agent.
        """
        return self.client.create_agent(self.project_id, function_source_code, agent_function_name, name, memory, package_requirements, description, enable_binary_input, evaluation_feature_group_id, agent_input_schema, agent_output_schema, workflow_graph, agent_interface, included_modules, org_level_connectors, user_level_connectors, initialize_function_name, initialize_function_code)

    def generate_agent_code(self, prompt: str, fast_mode: bool = None):
        """
        Generates the code for defining an AI Agent

        Args:
            prompt (str): A natural language prompt which describes agent specification. Describe what the agent will do, what inputs it will expect, and what outputs it will give out
            fast_mode (bool): If True, runs a faster but slightly less accurate code generation pipeline
        """
        return self.client.generate_agent_code(self.project_id, prompt, fast_mode)

    def list_agents(self):
        """
        Retrieves the list of agents in the specified project.

        Args:
            project_id (str): The unique identifier associated with the project.

        Returns:
            list[Agent]: A list of agents in the project.
        """
        return self.client.list_agents(self.project_id)

    def create_document_retriever(self, name: str, feature_group_id: str, document_retriever_config: Union[dict, VectorStoreConfig] = None):
        """
        Returns a document retriever that stores embeddings for document chunks in a feature group.

        Document columns in the feature group are broken into chunks. For cases with multiple document columns, chunks from all columns are combined together to form a single chunk.


        Args:
            name (str): The name of the Document Retriever. Can be up to 120 characters long and can only contain alphanumeric characters and underscores.
            feature_group_id (str): The ID of the feature group that the Document Retriever is associated with.
            document_retriever_config (VectorStoreConfig): The configuration, including chunk_size and chunk_overlap_fraction, for document retrieval.

        Returns:
            DocumentRetriever: The newly created document retriever.
        """
        return self.client.create_document_retriever(self.project_id, name, feature_group_id, document_retriever_config)

    def list_document_retrievers(self, limit: int = 100, start_after_id: str = None):
        """
        List all the document retrievers.

        Args:
            limit (int): The number of document retrievers to return.
            start_after_id (str): An offset parameter to exclude all document retrievers up to this specified ID.

        Returns:
            list[DocumentRetriever]: All the document retrievers in the organization associated with the specified project.
        """
        return self.client.list_document_retrievers(self.project_id, limit, start_after_id)

    def create_model_from_functions(self, train_function: callable, predict_function: callable = None, training_input_tables: list = None, predict_many_function: callable = None, initialize_function: callable = None, cpu_size: str = None, memory: int = None, training_config: dict = None, exclusive_run: bool = False):
        """
        Creates a model using python.

        Args:
            train_function (callable): The train function is passed.
            predict_function (callable): The prediction function is passed.
            training_input_tables (list): The input tables to be used for training the model. Defaults to None.
            predict_many_function (callable): Prediction function for batch input
            cpu_size (str): Size of the cpu for the feature group function
            memory (int): Memory (in GB) for the feature group function

        Returns:
            Model: The model object.
        """
        return self.client.create_model_from_functions(project_id=self.id, train_function=train_function, predict_function=predict_function, training_input_tables=training_input_tables, predict_many_function=predict_many_function, initialize_function=initialize_function, training_config=training_config, cpu_size=cpu_size, memory=memory, exclusive_run=exclusive_run)
