import time

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
            trainingInputTables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            sourceCode (str): Python code used to make the model.
            cpuSize (str): Cpu size specified for the python model training.
            memory (int): Memory in GB specified for the python model training.
            latestModelVersion (ModelVersion): The latest model version.
            location (ModelLocation): Location information for models that are imported.
            refreshSchedules (RefreshSchedule): List of refresh schedules that indicate when the next model version will be trained
    """

    def __init__(self, client, name=None, modelId=None, modelConfig=None, modelPredictionConfig=None, createdAt=None, projectId=None, shared=None, sharedAt=None, trainFunctionName=None, predictFunctionName=None, trainingInputTables=None, sourceCode=None, cpuSize=None, memory=None, location={}, refreshSchedules={}, latestModelVersion={}):
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
        self.training_input_tables = trainingInputTables
        self.source_code = sourceCode
        self.cpu_size = cpuSize
        self.memory = memory
        self.location = client._build_class(ModelLocation, location)
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)
        self.latest_model_version = client._build_class(
            ModelVersion, latestModelVersion)

    def __repr__(self):
        return f"Model(name={repr(self.name)},\n  model_id={repr(self.model_id)},\n  model_config={repr(self.model_config)},\n  model_prediction_config={repr(self.model_prediction_config)},\n  created_at={repr(self.created_at)},\n  project_id={repr(self.project_id)},\n  shared={repr(self.shared)},\n  shared_at={repr(self.shared_at)},\n  train_function_name={repr(self.train_function_name)},\n  predict_function_name={repr(self.predict_function_name)},\n  training_input_tables={repr(self.training_input_tables)},\n  source_code={repr(self.source_code)},\n  cpu_size={repr(self.cpu_size)},\n  memory={repr(self.memory)},\n  location={repr(self.location)},\n  refresh_schedules={repr(self.refresh_schedules)},\n  latest_model_version={repr(self.latest_model_version)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'model_id': self.model_id, 'model_config': self.model_config, 'model_prediction_config': self.model_prediction_config, 'created_at': self.created_at, 'project_id': self.project_id, 'shared': self.shared, 'shared_at': self.shared_at, 'train_function_name': self.train_function_name, 'predict_function_name': self.predict_function_name, 'training_input_tables': self.training_input_tables, 'source_code': self.source_code, 'cpu_size': self.cpu_size, 'memory': self.memory, 'location': self._get_attribute_as_dict(self.location), 'refresh_schedules': self._get_attribute_as_dict(self.refresh_schedules), 'latest_model_version': self._get_attribute_as_dict(self.latest_model_version)}

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

    def update_python(self, function_source_code: str = None, train_function_name: str = None, predict_function_name: str = None, training_input_tables: list = None, cpu_size: str = None, memory: int = None):
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
            training_input_tables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            cpu_size (str): Size of the cpu for the model training function
            memory (int): Memory (in GB) for the model training function

        Returns:
            Model: The updated model
        """
        return self.client.update_python_model(self.model_id, function_source_code, train_function_name, predict_function_name, training_input_tables, cpu_size, memory)

    def set_training_config(self, training_config: dict):
        """
        Edits the default model training config

        Args:
            training_config (dict): The training config key/value pairs used to train this model.

        Returns:
            Model: The model object correspoding after the training config is applied
        """
        return self.client.set_model_training_config(self.model_id, training_config)

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

    def retrain(self, deployment_ids: list = []):
        """
        Retrains the specified model. Gives you an option to choose the deployments you want the retraining to be deployed to.

        Args:
            deployment_ids (list): List of deployments to automatically deploy to.

        Returns:
            Model: The model that is being retrained.
        """
        return self.client.retrain_model(self.model_id, deployment_ids)

    def delete(self):
        """
        Deletes the specified model and all its versions. Models which are currently used in deployments cannot be deleted.

        Args:
            model_id (str): The ID of the model to delete.
        """
        return self.client.delete_model(self.model_id)

    def wait_for_training(self, timeout=None):
        """
        A waiting call until model is trained.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'TRAINING'}, delay=30, timeout=timeout)

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
        start_time = time.time()
        while True:
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f'Maximum wait time of {timeout}s exceeded')
            model_version = self.client._call_api('describeModel', 'GET', query_params={
                                                  'modelId': self.model_id, 'waitForFullAutoml': True}, parse_type=Model).latest_model_version
            if model_version.status not in {'PENDING', 'TRAINING'} and not model_version.pending_deployment_ids:
                break
            time.sleep(30)
        return self.describe()

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
