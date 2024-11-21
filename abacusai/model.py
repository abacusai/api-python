from typing import Dict, List, Union

from . import api_class
from .api_class import AgentInterface, TrainingConfig, WorkflowGraph
from .code_source import CodeSource
from .database_connector import DatabaseConnector
from .feature_group import FeatureGroup
from .model_location import ModelLocation
from .model_version import ModelVersion
from .refresh_schedule import RefreshSchedule
from .return_class import AbstractApiClass


class Model(AbstractApiClass):
    """
        A model

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The user-friendly name for the model.
            modelId (str): The unique identifier of the model.
            modelConfigType (str): Name of the TrainingConfig class of the model_config.
            modelPredictionConfig (dict): The prediction config options for the model.
            createdAt (str): Date and time at which the model was created.
            projectId (str): The project this model belongs to.
            shared (bool): If model is shared to the Abacus.AI model showcase.
            sharedAt (str): The date and time at which the model was shared to the model showcase
            trainFunctionName (str): Name of the function found in the source code that will be executed to train the model. It is not executed when this function is run.
            predictFunctionName (str): Name of the function found in the source code that will be executed run predictions through model. It is not executed when this function is run.
            predictManyFunctionName (str): Name of the function found in the source code that will be executed to run batch predictions trhough the model.
            initializeFunctionName (str): Name of the function found in the source code to initialize the trained model before using it to make predictions using the model
            trainingInputTables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            sourceCode (str): Python code used to make the model.
            cpuSize (str): Cpu size specified for the python model training.
            memory (int): Memory in GB specified for the python model training.
            trainingFeatureGroupIds (list of unique string identifiers): The unique identifiers of the feature groups used as the inputs to train this model on.
            algorithmModelConfigs (list[dict]): List of algorithm specific training configs.
            trainingVectorStoreVersions (list): The vector store version IDs used as inputs during training to create this ModelVersion.
            documentRetrievers (list): List of document retrievers use to create this model.
            documentRetrieverIds (list): List of document retriever IDs used to create this model.
            isPythonModel (bool): If this model is handled as python model
            defaultAlgorithm (str): If set, this algorithm will always be used when deploying the model regardless of the model metrics
            customAlgorithmConfigs (dict): User-defined configs for each of the user-defined custom algorithm
            restrictedAlgorithms (dict): User-selected algorithms to train.
            useGpu (bool): If this model uses gpu.
            notebookId (str): The notebook associated with this model.
            trainingRequired (bool): If training is required to keep the model up-to-date.
            latestModelVersion (ModelVersion): The latest model version.
            location (ModelLocation): Location information for models that are imported.
            refreshSchedules (RefreshSchedule): List of refresh schedules that indicate when the next model version will be trained
            codeSource (CodeSource): If a python model, information on the source code
            databaseConnector (DatabaseConnector): Database connector used by the model.
            dataLlmFeatureGroups (FeatureGroup): List of feature groups used by the model for queries
            modelConfig (TrainingConfig): The training config options used to train this model.
    """

    def __init__(self, client, name=None, modelId=None, modelConfigType=None, modelPredictionConfig=None, createdAt=None, projectId=None, shared=None, sharedAt=None, trainFunctionName=None, predictFunctionName=None, predictManyFunctionName=None, initializeFunctionName=None, trainingInputTables=None, sourceCode=None, cpuSize=None, memory=None, trainingFeatureGroupIds=None, algorithmModelConfigs=None, trainingVectorStoreVersions=None, documentRetrievers=None, documentRetrieverIds=None, isPythonModel=None, defaultAlgorithm=None, customAlgorithmConfigs=None, restrictedAlgorithms=None, useGpu=None, notebookId=None, trainingRequired=None, location={}, refreshSchedules={}, codeSource={}, databaseConnector={}, dataLlmFeatureGroups={}, latestModelVersion={}, modelConfig={}):
        super().__init__(client, modelId)
        self.name = name
        self.model_id = modelId
        self.model_config_type = modelConfigType
        self.model_prediction_config = modelPredictionConfig
        self.created_at = createdAt
        self.project_id = projectId
        self.shared = shared
        self.shared_at = sharedAt
        self.train_function_name = trainFunctionName
        self.predict_function_name = predictFunctionName
        self.predict_many_function_name = predictManyFunctionName
        self.initialize_function_name = initializeFunctionName
        self.training_input_tables = trainingInputTables
        self.source_code = sourceCode
        self.cpu_size = cpuSize
        self.memory = memory
        self.training_feature_group_ids = trainingFeatureGroupIds
        self.algorithm_model_configs = algorithmModelConfigs
        self.training_vector_store_versions = trainingVectorStoreVersions
        self.document_retrievers = documentRetrievers
        self.document_retriever_ids = documentRetrieverIds
        self.is_python_model = isPythonModel
        self.default_algorithm = defaultAlgorithm
        self.custom_algorithm_configs = customAlgorithmConfigs
        self.restricted_algorithms = restrictedAlgorithms
        self.use_gpu = useGpu
        self.notebook_id = notebookId
        self.training_required = trainingRequired
        self.location = client._build_class(ModelLocation, location)
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)
        self.code_source = client._build_class(CodeSource, codeSource)
        self.database_connector = client._build_class(
            DatabaseConnector, databaseConnector)
        self.data_llm_feature_groups = client._build_class(
            FeatureGroup, dataLlmFeatureGroups)
        self.latest_model_version = client._build_class(
            ModelVersion, latestModelVersion)
        self.model_config = client._build_class(getattr(
            api_class, modelConfigType, TrainingConfig) if modelConfigType else TrainingConfig, modelConfig)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'model_id': repr(self.model_id), f'model_config_type': repr(self.model_config_type), f'model_prediction_config': repr(self.model_prediction_config), f'created_at': repr(self.created_at), f'project_id': repr(self.project_id), f'shared': repr(self.shared), f'shared_at': repr(self.shared_at), f'train_function_name': repr(self.train_function_name), f'predict_function_name': repr(self.predict_function_name), f'predict_many_function_name': repr(self.predict_many_function_name), f'initialize_function_name': repr(self.initialize_function_name), f'training_input_tables': repr(self.training_input_tables), f'source_code': repr(self.source_code), f'cpu_size': repr(self.cpu_size), f'memory': repr(self.memory), f'training_feature_group_ids': repr(self.training_feature_group_ids), f'algorithm_model_configs': repr(self.algorithm_model_configs),
                     f'training_vector_store_versions': repr(self.training_vector_store_versions), f'document_retrievers': repr(self.document_retrievers), f'document_retriever_ids': repr(self.document_retriever_ids), f'is_python_model': repr(self.is_python_model), f'default_algorithm': repr(self.default_algorithm), f'custom_algorithm_configs': repr(self.custom_algorithm_configs), f'restricted_algorithms': repr(self.restricted_algorithms), f'use_gpu': repr(self.use_gpu), f'notebook_id': repr(self.notebook_id), f'training_required': repr(self.training_required), f'location': repr(self.location), f'refresh_schedules': repr(self.refresh_schedules), f'code_source': repr(self.code_source), f'database_connector': repr(self.database_connector), f'data_llm_feature_groups': repr(self.data_llm_feature_groups), f'latest_model_version': repr(self.latest_model_version), f'model_config': repr(self.model_config)}
        class_name = "Model"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'model_id': self.model_id, 'model_config_type': self.model_config_type, 'model_prediction_config': self.model_prediction_config, 'created_at': self.created_at, 'project_id': self.project_id, 'shared': self.shared, 'shared_at': self.shared_at, 'train_function_name': self.train_function_name, 'predict_function_name': self.predict_function_name, 'predict_many_function_name': self.predict_many_function_name, 'initialize_function_name': self.initialize_function_name, 'training_input_tables': self.training_input_tables, 'source_code': self.source_code, 'cpu_size': self.cpu_size, 'memory': self.memory, 'training_feature_group_ids': self.training_feature_group_ids, 'algorithm_model_configs': self.algorithm_model_configs, 'training_vector_store_versions': self.training_vector_store_versions, 'document_retrievers': self.document_retrievers,
                'document_retriever_ids': self.document_retriever_ids, 'is_python_model': self.is_python_model, 'default_algorithm': self.default_algorithm, 'custom_algorithm_configs': self.custom_algorithm_configs, 'restricted_algorithms': self.restricted_algorithms, 'use_gpu': self.use_gpu, 'notebook_id': self.notebook_id, 'training_required': self.training_required, 'location': self._get_attribute_as_dict(self.location), 'refresh_schedules': self._get_attribute_as_dict(self.refresh_schedules), 'code_source': self._get_attribute_as_dict(self.code_source), 'database_connector': self._get_attribute_as_dict(self.database_connector), 'data_llm_feature_groups': self._get_attribute_as_dict(self.data_llm_feature_groups), 'latest_model_version': self._get_attribute_as_dict(self.latest_model_version), 'model_config': self._get_attribute_as_dict(self.model_config)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def describe_train_test_data_split_feature_group(self):
        """
        Get the train and test data split for a trained model by its unique identifier. This is only supported for models with custom algorithms.

        Args:
            model_id (str): The unique ID of the model. By default, the latest model version will be returned if no version is specified.

        Returns:
            FeatureGroup: The feature group containing the training data and fold information.
        """
        return self.client.describe_train_test_data_split_feature_group(self.model_id)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            Model: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Retrieves a full description of the specified model.

        Args:
            model_id (str): Unique string identifier associated with the model.

        Returns:
            Model: Description of the model.
        """
        return self.client.describe_model(self.model_id)

    def rename(self, name: str):
        """
        Renames a model

        Args:
            name (str): The new name to assign to the model.
        """
        return self.client.rename_model(self.model_id, name)

    def update_python(self, function_source_code: str = None, train_function_name: str = None, predict_function_name: str = None, predict_many_function_name: str = None, initialize_function_name: str = None, training_input_tables: list = None, cpu_size: str = None, memory: int = None, package_requirements: list = None, use_gpu: bool = None, is_thread_safe: bool = None, training_config: Union[dict, TrainingConfig] = None):
        """
        Updates an existing Python Model using user-provided Python code. If a list of input feature groups is supplied, they will be provided as arguments to the `train` and `predict` functions with the materialized feature groups for those input feature groups.

        This method expects `functionSourceCode` to be a valid language source file which contains the functions named `trainFunctionName` and `predictFunctionName`. `trainFunctionName` returns the ModelVersion that is the result of training the model using `trainFunctionName`. `predictFunctionName` has no well-defined return type, as it returns the prediction made by the `predictFunctionName`, which can be anything.


        Args:
            function_source_code (str): Contents of a valid Python source code file. The source code should contain the functions named `trainFunctionName` and `predictFunctionName`. A list of allowed import and system libraries for each language is specified in the user functions documentation section.
            train_function_name (str): Name of the function found in the source code that will be executed to train the model. It is not executed when this function is run.
            predict_function_name (str): Name of the function found in the source code that will be executed to run predictions through the model. It is not executed when this function is run.
            predict_many_function_name (str): Name of the function found in the source code that will be executed to run batch predictions through the model. It is not executed when this function is run.
            initialize_function_name (str): Name of the function found in the source code to initialize the trained model before using it to make predictions using the model.
            training_input_tables (list): List of feature groups that are supplied to the `train` function as parameters. Each of the parameters are materialized DataFrames (same type as the functions return value).
            cpu_size (str): Size of the CPU for the model training function.
            memory (int): Memory (in GB) for the model training function.
            package_requirements (list): List of package requirement strings. For example: `['numpy==1.2.3', 'pandas>=1.4.0']`.
            use_gpu (bool): Whether this model needs gpu
            is_thread_safe (bool): Whether this model is thread safe
            training_config (TrainingConfig): The training config used to train this model.

        Returns:
            Model: The updated model.
        """
        return self.client.update_python_model(self.model_id, function_source_code, train_function_name, predict_function_name, predict_many_function_name, initialize_function_name, training_input_tables, cpu_size, memory, package_requirements, use_gpu, is_thread_safe, training_config)

    def update_python_zip(self, train_function_name: str = None, predict_function_name: str = None, predict_many_function_name: str = None, train_module_name: str = None, predict_module_name: str = None, training_input_tables: list = None, cpu_size: str = None, memory: int = None, package_requirements: list = None, use_gpu: bool = None):
        """
        Updates an existing Python Model using a provided zip file. If a list of input feature groups are supplied, they will be provided as arguments to the train and predict functions with the materialized feature groups for those input feature groups.

        This method expects `trainModuleName` and `predictModuleName` to be valid language source files which contain the functions named `trainFunctionName` and `predictFunctionName`, respectively. `trainFunctionName` returns the ModelVersion that is the result of training the model using `trainFunctionName`, and `predictFunctionName` has no well-defined return type, as it returns the prediction made by the `predictFunctionName`, which can be anything.


        Args:
            train_function_name (str): Name of the function found in the train module that will be executed to train the model. It is not executed when this function is run.
            predict_function_name (str): Name of the function found in the predict module that will be executed to run predictions through the model. It is not executed when this function is run.
            predict_many_function_name (str): Name of the function found in the predict module that will be executed to run batch predictions through the model. It is not executed when this function is run.
            train_module_name (str): Full path of the module that contains the train function from the root of the zip.
            predict_module_name (str): Full path of the module that contains the predict function from the root of the zip.
            training_input_tables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the function's return value).
            cpu_size (str): Size of the CPU for the model training function.
            memory (int): Memory (in GB) for the model training function.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            use_gpu (bool): Whether this model needs gpu

        Returns:
            Upload: The updated model.
        """
        return self.client.update_python_model_zip(self.model_id, train_function_name, predict_function_name, predict_many_function_name, train_module_name, predict_module_name, training_input_tables, cpu_size, memory, package_requirements, use_gpu)

    def update_python_git(self, application_connector_id: str = None, branch_name: str = None, python_root: str = None, train_function_name: str = None, predict_function_name: str = None, predict_many_function_name: str = None, train_module_name: str = None, predict_module_name: str = None, training_input_tables: list = None, cpu_size: str = None, memory: int = None, use_gpu: bool = None):
        """
        Updates an existing Python model using an existing Git application connector. If a list of input feature groups are supplied, these will be provided as arguments to the train and predict functions with the materialized feature groups for those input feature groups.

        This method expects `trainModuleName` and `predictModuleName` to be valid language source files which contain the functions named `trainFunctionName` and `predictFunctionName`, respectively. `trainFunctionName` returns the `ModelVersion` that is the result of training the model using `trainFunctionName`, and `predictFunctionName` has no well-defined return type, as it returns the prediction made by the `predictFunctionName`, which can be anything.


        Args:
            application_connector_id (str): The unique ID associated with the Git application connector.
            branch_name (str): Name of the branch in the Git repository to be used for training.
            python_root (str): Path from the top level of the Git repository to the directory containing the Python source code. If not provided, the default is the root of the Git repository.
            train_function_name (str): Name of the function found in train module that will be executed to train the model. It is not executed when this function is run.
            predict_function_name (str): Name of the function found in the predict module that will be executed to run predictions through model. It is not executed when this function is run.
            predict_many_function_name (str): Name of the function found in the predict module that will be executed to run batch predictions through model. It is not executed when this function is run.
            train_module_name (str): Full path of the module that contains the train function from the root of the zip.
            predict_module_name (str): Full path of the module that contains the predict function from the root of the zip.
            training_input_tables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            cpu_size (str): Size of the CPU for the model training function.
            memory (int): Memory (in GB) for the model training function.
            use_gpu (bool): Whether this model needs gpu

        Returns:
            Model: The updated model.
        """
        return self.client.update_python_model_git(self.model_id, application_connector_id, branch_name, python_root, train_function_name, predict_function_name, predict_many_function_name, train_module_name, predict_module_name, training_input_tables, cpu_size, memory, use_gpu)

    def set_training_config(self, training_config: Union[dict, TrainingConfig], feature_group_ids: List = None):
        """
        Edits the default model training config

        Args:
            training_config (TrainingConfig): The training config used to train this model.
            feature_group_ids (List): The list of feature groups used as input to the model.

        Returns:
            Model: The model object corresponding to the updated training config.
        """
        return self.client.set_model_training_config(self.model_id, training_config, feature_group_ids)

    def set_prediction_params(self, prediction_config: dict):
        """
        Sets the model prediction config for the model

        Args:
            prediction_config (dict): Prediction configuration for the model.

        Returns:
            Model: Model object after the prediction configuration is applied.
        """
        return self.client.set_model_prediction_params(self.model_id, prediction_config)

    def get_metrics(self, model_version: str = None, return_graphs: bool = False, validation: bool = False):
        """
        Retrieves metrics for all the algorithms trained in this model version.

        If only the model's unique identifier (model_id) is specified, the latest trained version of the model (model_version) is used.


        Args:
            model_version (str): Version of the model.
            return_graphs (bool): If true, will return the information used for the graphs on the model metrics page such as PR Curve per label.
            validation (bool): If true, will return the validation metrics instead of the test metrics.

        Returns:
            ModelMetrics: An object containing the model metrics and explanations for what each metric means.
        """
        return self.client.get_model_metrics(self.model_id, model_version, return_graphs, validation)

    def list_versions(self, limit: int = 100, start_after_version: str = None):
        """
        Retrieves a list of versions for a given model.

        Args:
            limit (int): Maximum length of the list of all dataset versions.
            start_after_version (str): Unique string identifier of the version after which the list starts.

        Returns:
            list[ModelVersion]: An array of model versions.
        """
        return self.client.list_model_versions(self.model_id, limit, start_after_version)

    def retrain(self, deployment_ids: List = None, feature_group_ids: List = None, custom_algorithms: list = None, builtin_algorithms: list = None, custom_algorithm_configs: dict = None, cpu_size: str = None, memory: int = None, training_config: Union[dict, TrainingConfig] = None, algorithm_training_configs: list = None):
        """
        Retrains the specified model, with an option to choose the deployments to which the retraining will be deployed.

        Args:
            deployment_ids (List): List of unique string identifiers of deployments to automatically deploy to.
            feature_group_ids (List): List of feature group IDs provided by the user to train the model on.
            custom_algorithms (list): List of user-defined algorithms to train. If not set, will honor the runs from the last time and applicable new custom algorithms.
            builtin_algorithms (list): List of algorithm names or algorithm IDs of Abacus.AI built-in algorithms to train. If not set, will honor the runs from the last time and applicable new built-in algorithms.
            custom_algorithm_configs (dict): User-defined training configs for each custom algorithm.
            cpu_size (str): Size of the CPU for the user-defined algorithms during training.
            memory (int): Memory (in GB) for the user-defined algorithms during training.
            training_config (TrainingConfig): The training config used to train this model.
            algorithm_training_configs (list): List of algorithm specifc training configs that will be part of the model training AutoML run.

        Returns:
            Model: The model that is being retrained.
        """
        return self.client.retrain_model(self.model_id, deployment_ids, feature_group_ids, custom_algorithms, builtin_algorithms, custom_algorithm_configs, cpu_size, memory, training_config, algorithm_training_configs)

    def delete(self):
        """
        Deletes the specified model and all its versions. Models which are currently used in deployments cannot be deleted.

        Args:
            model_id (str): Unique string identifier of the model to delete.
        """
        return self.client.delete_model(self.model_id)

    def set_default_algorithm(self, algorithm: str = None, data_cluster_type: str = None):
        """
        Sets the model's algorithm to default for all new deployments

        Args:
            algorithm (str): Algorithm to pin in the model.
            data_cluster_type (str): Data cluster type to set the lead model for.
        """
        return self.client.set_default_model_algorithm(self.model_id, algorithm, data_cluster_type)

    def list_artifacts_exports(self, limit: int = 25):
        """
        List all the model artifacts exports.

        Args:
            limit (int): Maximum length of the list of all exports.

        Returns:
            list[ModelArtifactsExport]: List of model artifacts exports.
        """
        return self.client.list_model_artifacts_exports(self.model_id, limit)

    def get_training_types_for_deployment(self, model_version: str = None, algorithm: str = None):
        """
        Returns types of models that can be deployed for a given model instance ID.

        Args:
            model_version (str): The unique ID associated with the model version to deploy.
            algorithm (str): The unique ID associated with the algorithm to deploy.

        Returns:
            ModelTrainingTypeForDeployment: Model training types for deployment.
        """
        return self.client.get_model_training_types_for_deployment(self.model_id, model_version, algorithm)

    def update_agent(self, function_source_code: str = None, agent_function_name: str = None, memory: int = None, package_requirements: list = None, description: str = None, enable_binary_input: bool = None, agent_input_schema: dict = None, agent_output_schema: dict = None, workflow_graph: Union[dict, WorkflowGraph] = None, agent_interface: Union[dict, AgentInterface] = None, included_modules: List = None, org_level_connectors: List = None, user_level_connectors: Dict = None, initialize_function_name: str = None, initialize_function_code: str = None):
        """
        Updates an existing AI Agent. A new version of the agent will be created and published.

        Args:
            memory (int): Memory (in GB) for the agent.
            package_requirements (list): A list of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            description (str): A description of the agent, including its purpose and instructions.
            workflow_graph (WorkflowGraph): The workflow graph for the agent.
            agent_interface (AgentInterface): The interface that the agent will be deployed with.
            included_modules (List): A list of user created custom modules to include in the agent's environment.
            org_level_connectors (List): A list of org level connector ids to be used by the agent.
            user_level_connectors (Dict): A dictionary mapping ApplicationConnectorType keys to lists of OAuth scopes. Each key represents a specific user level application connector, while the value is a list of scopes that define the permissions granted to the application.
            initialize_function_name (str): The name of the function to be used for initialization.
            initialize_function_code (str): The function code to be used for initialization.

        Returns:
            Agent: The updated agent.
        """
        return self.client.update_agent(self.model_id, function_source_code, agent_function_name, memory, package_requirements, description, enable_binary_input, agent_input_schema, agent_output_schema, workflow_graph, agent_interface, included_modules, org_level_connectors, user_level_connectors, initialize_function_name, initialize_function_code)

    def wait_for_training(self, timeout=None):
        """
        A waiting call until model is trained.

        Args:
            timeout (int): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        latest_model_version = self.describe().latest_model_version
        if not latest_model_version:
            from .client import ApiException
            raise ApiException(409, 'This model does not have any versions')
        self.latest_model_version = latest_model_version.wait_for_training(
            timeout=timeout)
        return self

    def wait_for_evaluation(self, timeout=None):
        """
        A waiting call until model is evaluated completely.

        Args:
            timeout (int): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.wait_for_training()

    def wait_for_publish(self, timeout=None):
        """
        A waiting call until agent is published.

        Args:
            timeout (int): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.wait_for_training()

    def wait_for_full_automl(self, timeout=None):
        """
        A waiting call until full AutoML cycle is completed.

        Args:
            timeout (int): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        latest_model_version = self.describe().latest_model_version
        if not latest_model_version:
            from .client import ApiException
            raise ApiException(409, 'This model does not have any versions')
        self.latest_model_version = latest_model_version.wait_for_full_automl(
            timeout=timeout)
        return self

    def get_status(self, get_automl_status: bool = False):
        """
        Gets the status of the model training.

        Returns:
            str: A string describing the status of a model training (pending, complete, etc.).
        """
        if get_automl_status:
            return self.client._call_api('describeModel', 'GET', query_params={'modelId': self.model_id, 'waitForFullAutoml': True}, parse_type=Model).latest_model_version.status
        return self.describe().latest_model_version.status

    def create_refresh_policy(self, cron: str):
        """
        To create a refresh policy for a model.

        Args:
            cron (str): A cron style string to set the refresh time.

        Returns:
            RefreshPolicy: The refresh policy object.
        """
        return self.client.create_refresh_policy(self.name, cron, 'MODEL', model_ids=[self.id])

    def list_refresh_policies(self):
        """
        Gets the refresh policies in a list.

        Returns:
            List[RefreshPolicy]: A list of refresh policy objects.
        """
        return self.client.list_refresh_policies(model_ids=[self.id])

    def get_train_test_feature_group_as_pandas(self):
        """
        Get the model train test data split feature group as pandas.

        Returns:
            pandas.Dataframe: A pandas dataframe for the training data with fold column.
        """
        return self.client.describe_train_test_data_split_feature_group(self.model_id).load_as_pandas()
