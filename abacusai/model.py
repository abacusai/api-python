from .code_source import CodeSource
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
            modelConfig (dict): The training config options used to train this model.
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
            isPythonModel (bool): If this model is handled as python model
            defaultAlgorithm (str): If set, this algorithm will always be used when deploying the model regardless of the model metrics
            customAlgorithmConfigs (dict): User-defined configs for each of the user-defined custom algorithm
            restrictedAlgorithms (dict): User-selected algorithms to train.
            latestModelVersion (ModelVersion): The latest model version.
            location (ModelLocation): Location information for models that are imported.
            refreshSchedules (RefreshSchedule): List of refresh schedules that indicate when the next model version will be trained
            codeSource (CodeSource): If a python model, information on the source code
    """

    def __init__(self, client, name=None, modelId=None, modelConfig=None, modelPredictionConfig=None, createdAt=None, projectId=None, shared=None, sharedAt=None, trainFunctionName=None, predictFunctionName=None, predictManyFunctionName=None, initializeFunctionName=None, trainingInputTables=None, sourceCode=None, cpuSize=None, memory=None, trainingFeatureGroupIds=None, isPythonModel=None, defaultAlgorithm=None, customAlgorithmConfigs=None, restrictedAlgorithms=None, location={}, refreshSchedules={}, codeSource={}, latestModelVersion={}):
        super().__init__(client, modelId)
        self.name = name
        self.model_id = modelId
        self.model_config = modelConfig
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
        self.is_python_model = isPythonModel
        self.default_algorithm = defaultAlgorithm
        self.custom_algorithm_configs = customAlgorithmConfigs
        self.restricted_algorithms = restrictedAlgorithms
        self.location = client._build_class(ModelLocation, location)
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)
        self.code_source = client._build_class(CodeSource, codeSource)
        self.latest_model_version = client._build_class(
            ModelVersion, latestModelVersion)

    def __repr__(self):
        return f"Model(name={repr(self.name)},\n  model_id={repr(self.model_id)},\n  model_config={repr(self.model_config)},\n  model_prediction_config={repr(self.model_prediction_config)},\n  created_at={repr(self.created_at)},\n  project_id={repr(self.project_id)},\n  shared={repr(self.shared)},\n  shared_at={repr(self.shared_at)},\n  train_function_name={repr(self.train_function_name)},\n  predict_function_name={repr(self.predict_function_name)},\n  predict_many_function_name={repr(self.predict_many_function_name)},\n  initialize_function_name={repr(self.initialize_function_name)},\n  training_input_tables={repr(self.training_input_tables)},\n  source_code={repr(self.source_code)},\n  cpu_size={repr(self.cpu_size)},\n  memory={repr(self.memory)},\n  training_feature_group_ids={repr(self.training_feature_group_ids)},\n  is_python_model={repr(self.is_python_model)},\n  default_algorithm={repr(self.default_algorithm)},\n  custom_algorithm_configs={repr(self.custom_algorithm_configs)},\n  restricted_algorithms={repr(self.restricted_algorithms)},\n  location={repr(self.location)},\n  refresh_schedules={repr(self.refresh_schedules)},\n  code_source={repr(self.code_source)},\n  latest_model_version={repr(self.latest_model_version)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'model_id': self.model_id, 'model_config': self.model_config, 'model_prediction_config': self.model_prediction_config, 'created_at': self.created_at, 'project_id': self.project_id, 'shared': self.shared, 'shared_at': self.shared_at, 'train_function_name': self.train_function_name, 'predict_function_name': self.predict_function_name, 'predict_many_function_name': self.predict_many_function_name, 'initialize_function_name': self.initialize_function_name, 'training_input_tables': self.training_input_tables, 'source_code': self.source_code, 'cpu_size': self.cpu_size, 'memory': self.memory, 'training_feature_group_ids': self.training_feature_group_ids, 'is_python_model': self.is_python_model, 'default_algorithm': self.default_algorithm, 'custom_algorithm_configs': self.custom_algorithm_configs, 'restricted_algorithms': self.restricted_algorithms, 'location': self._get_attribute_as_dict(self.location), 'refresh_schedules': self._get_attribute_as_dict(self.refresh_schedules), 'code_source': self._get_attribute_as_dict(self.code_source), 'latest_model_version': self._get_attribute_as_dict(self.latest_model_version)}

    def describe_train_test_data_split_feature_group(self):
        """
        Get the train and test data split for a trained model by model id. Only supported for models with custom algorithms.

        Args:
            model_id (str): The unique ID of the model. By default will return for latest model version if version is not specified.

        Returns:
            FeatureGroup: The feature group containing the training data and folds information.
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
            model_id (str): The unique ID associated with the model.

        Returns:
            Model: The description of the model.
        """
        return self.client.describe_model(self.model_id)

    def rename(self, name: str):
        """
        Renames a model

        Args:
            name (str): The name to apply to the model
        """
        return self.client.rename_model(self.model_id, name)

    def update_python(self, function_source_code: str = None, train_function_name: str = None, predict_function_name: str = None, predict_many_function_name: str = None, initialize_function_name: str = None, training_input_tables: list = None, cpu_size: str = None, memory: int = None, package_requirements: dict = None):
        """
        Updates an existing python Model using user provided Python code. If a list of input feature groups are supplied,

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
            predict_many_function_name (str): Name of the function found in the source code that will be executed to run batch predictions through model. It is not executed when this function is run.
            initialize_function_name (str): Name of the function found in the source code to initialize the trained model before using it to make predictions using the model
            training_input_tables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            cpu_size (str): Size of the cpu for the model training function
            memory (int): Memory (in GB) for the model training function
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency

        Returns:
            Model: The updated model
        """
        return self.client.update_python_model(self.model_id, function_source_code, train_function_name, predict_function_name, predict_many_function_name, initialize_function_name, training_input_tables, cpu_size, memory, package_requirements)

    def update_python_zip(self, train_function_name: str = None, predict_function_name: str = None, predict_many_function_name: str = None, train_module_name: str = None, predict_module_name: str = None, training_input_tables: list = None, cpu_size: str = None, memory: int = None, package_requirements: dict = None):
        """
        Updates an existing python Model using a provided zip file. If a list of input feature groups are supplied,

        we will provide as arguments to the train and predict functions with the materialized feature groups for those
        input feature groups.

        This method expects `trainModuleName` and `predictModuleName` to be valid language source files which contains the functions named
        `trainFunctionName` and `predictFunctionName`, respectively. `trainFunctionName` returns the ModelVersion that is the result of
        training the model using `trainFunctionName` and `predictFunctionName` has no well defined return type,
        as it returns the prediction made by the `predictFunctionName`, which can be anything


        Args:
            train_function_name (str): Name of the function found in train module that will be executed to train the model. It is not executed when this function is run.
            predict_function_name (str): Name of the function found in the predict module that will be executed run predictions through model. It is not executed when this function is run.
            predict_many_function_name (str): Name of the function found in the predict module that will be executed run batch predictions through model. It is not executed when this function is run.
            train_module_name (str): Full path of the module that contains the train function from the root of the zip.
            predict_module_name (str): Full path of the module that contains the predict function from the root of the zip.
            training_input_tables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            cpu_size (str): Size of the cpu for the model training function
            memory (int): Memory (in GB) for the model training function
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency

        Returns:
            Upload: The updated model
        """
        return self.client.update_python_model_zip(self.model_id, train_function_name, predict_function_name, predict_many_function_name, train_module_name, predict_module_name, training_input_tables, cpu_size, memory, package_requirements)

    def update_python_git(self, application_connector_id: str = None, branch_name: str = None, python_root: str = None, train_function_name: str = None, predict_function_name: str = None, predict_many_function_name: str = None, train_module_name: str = None, predict_module_name: str = None, training_input_tables: list = None, cpu_size: str = None, memory: int = None):
        """
        Updates an existing python Model using an existing git application connector. If a list of input feature groups are supplied,

        we will provide as arguments to the train and predict functions with the materialized feature groups for those
        input feature groups.

        This method expects `trainModuleName` and `predictModuleName` to be valid language source files which contains the functions named
        `trainFunctionName` and `predictFunctionName`, respectively. `trainFunctionName` returns the ModelVersion that is the result of
        training the model using `trainFunctionName` and `predictFunctionName` has no well defined return type,
        as it returns the prediction made by the `predictFunctionName`, which can be anything


        Args:
            application_connector_id (str): The unique ID associated with the git application connector.
            branch_name (str): Name of the branch in the git repository to be used for training.
            python_root (str): Path from the top level of the git repository to the directory containing the Python source code. If not provided, the default is the root of the git repository.
            train_function_name (str): Name of the function found in train module that will be executed to train the model. It is not executed when this function is run.
            predict_function_name (str): Name of the function found in the predict module that will be executed run predictions through model. It is not executed when this function is run.
            predict_many_function_name (str): Name of the function found in the predict module that will be executed run batch predictions through model. It is not executed when this function is run.
            train_module_name (str): Full path of the module that contains the train function from the root of the zip.
            predict_module_name (str): Full path of the module that contains the predict function from the root of the zip.
            training_input_tables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            cpu_size (str): Size of the cpu for the model training function
            memory (int): Memory (in GB) for the model training function

        Returns:
            Model: The updated model
        """
        return self.client.update_python_model_git(self.model_id, application_connector_id, branch_name, python_root, train_function_name, predict_function_name, predict_many_function_name, train_module_name, predict_module_name, training_input_tables, cpu_size, memory)

    def set_training_config(self, training_config: dict, feature_group_ids: list = None):
        """
        Edits the default model training config

        Args:
            training_config (dict): The training config key/value pairs used to train this model.
            feature_group_ids (list): 

        Returns:
            Model: The model object correspoding after the training config is applied
        """
        return self.client.set_model_training_config(self.model_id, training_config, feature_group_ids)

    def set_prediction_params(self, prediction_config: dict):
        """
        Sets the model prediction config for the model

        Args:
            prediction_config (dict): The prediction config for the model

        Returns:
            Model: The model object correspoding after the prediction config is applied
        """
        return self.client.set_model_prediction_params(self.model_id, prediction_config)

    def get_metrics(self, model_version: str = None, baseline_metrics: bool = False):
        """
        Retrieves a full list of the metrics for the specified model.

        If only the model's unique identifier (modelId) is specified, the latest trained version of model (modelVersion) is used.


        Args:
            model_version (str): The version of the model.
            baseline_metrics (bool): If true, will also return the baseline model metrics for comparison.

        Returns:
            ModelMetrics: An object to show the model metrics and explanations for what each metric means.
        """
        return self.client.get_model_metrics(self.model_id, model_version, baseline_metrics)

    def list_versions(self, limit: int = 100, start_after_version: str = None):
        """
        Retrieves a list of the version for a given model.

        Args:
            limit (int): The max length of the list of all dataset versions.
            start_after_version (str): The id of the version after which the list starts.

        Returns:
            ModelVersion: An array of model versions.
        """
        return self.client.list_model_versions(self.model_id, limit, start_after_version)

    def retrain(self, deployment_ids: list = [], feature_group_ids: list = None, custom_algorithms: list = None, builtin_algorithms: list = None, custom_algorithm_configs: dict = None, cpu_size: str = None, memory: int = None, training_config: dict = None):
        """
        Retrains the specified model. Gives you an option to choose the deployments you want the retraining to be deployed to.

        Args:
            deployment_ids (list): List of deployments to automatically deploy to.
            feature_group_ids (list): List of feature group ids provided by the user to train the model on.
            custom_algorithms (list): List of user-defined algorithms to train. If not set, will honor the runs from last time and applicable new custom algorithms.
            builtin_algorithms (list): List of the builtin algorithms provided by Abacus.AI to train. If not set, honor the runs from last time and applicable new builtin algorithms.
            custom_algorithm_configs (dict): The user-defined training configs for each custom algorithm.
            cpu_size (str): Size of the cpu for the user-defined algorithms during train.
            memory (int): Memory (in GB) for the user-defined algorithms during train.
            training_config (dict): The training config key/value pairs used to train this model.

        Returns:
            Model: The model that is being retrained.
        """
        return self.client.retrain_model(self.model_id, deployment_ids, feature_group_ids, custom_algorithms, builtin_algorithms, custom_algorithm_configs, cpu_size, memory, training_config)

    def delete(self):
        """
        Deletes the specified model and all its versions. Models which are currently used in deployments cannot be deleted.

        Args:
            model_id (str): The ID of the model to delete.
        """
        return self.client.delete_model(self.model_id)

    def set_default_algorithm(self, algorithm: str = None, data_cluster_type: str = None):
        """
        Sets the model's algorithm to default for all new deployments

        Args:
            model_id (Unique String Identifier): The model to set
            algorithm (Enum String): the algorithm to pin in the model
            data_cluster_type (String): the data cluster type to set the lead model for


        Args:
            algorithm (str): 
            data_cluster_type (str): 
        """
        return self.client.set_default_model_algorithm(self.model_id, algorithm, data_cluster_type)

    def get_training_types_for_deployment(self, model_version: str = None, algorithm: str = None):
        """
        Returns types of models we can deploy for given model instance id

        Args:
            model_version (str): The unique ID associated with the model version to deploy.
            algorithm (str): The unique ID associated with the algorithm to deploy.

        Returns:
            ModelTrainingTypeForDeployment: Model training types for deployment
        """
        return self.client.get_model_training_types_for_deployment(self.model_id, model_version, algorithm)

    def wait_for_training(self, timeout=None):
        """
        A waiting call until model is trained.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
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
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.wait_for_training()

    def wait_for_full_automl(self, timeout=None):
        """
        A waiting call until full AutoML cycle is completed.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
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
