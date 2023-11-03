import time

from .api_class import DeployableAlgorithm, TrainingConfig
from .code_source import CodeSource
from .return_class import AbstractApiClass


class ModelVersion(AbstractApiClass):
    """
        A version of a model

        Args:
            client (ApiClient): An authenticated API Client instance
            modelVersion (str): The unique identifier of a model version.
            status (str): The current status of the model.
            modelId (str): A reference to the model this version belongs to.
            modelPredictionConfig (dict): The prediction config options for the model.
            trainingStartedAt (str): The start time and date of the training process in ISO-8601 format.
            trainingCompletedAt (str): The end time and date of the training process in ISO-8601 format.
            featureGroupVersions (list): A list of Feature Group version IDs used for model training.
            error (str): Relevant error if the status is FAILED.
            pendingDeploymentIds (list): List of deployment IDs where deployment is pending.
            failedDeploymentIds (list): List of failed deployment IDs.
            cpuSize (str): CPU size specified for the python model training.
            memory (int): Memory in GB specified for the python model training.
            automlComplete (bool): If true, all algorithms have completed training.
            trainingFeatureGroupIds (list): The unique identifiers of the feature groups used as inputs during training to create this ModelVersion.
            trainingVectorStoreVersions (list): The vector store version IDs used as inputs during training to create this ModelVersion.
            bestAlgorithm (dict): Best performing algorithm.
            defaultAlgorithm (dict): Default algorithm that the user has selected.
            featureAnalysisStatus (str): Lifecycle of the feature analysis stage.
            dataClusterInfo (dict): Information about the models for different data clusters.
            customAlgorithmConfigs (dict): User-defined configs for each of the user-defined custom algorithms.
            trainedModelTypes (list): List of trained model types.
            useGpu (bool): Whether this model version is using gpu
            partialComplete (bool): If true, all required algorithms have completed training.
            modelFeatureGroupSchemaMappings (dict): mapping of feature group to schema version
            codeSource (CodeSource): If a python model, information on where the source code is located.
    """

    def __init__(self, client, modelVersion=None, status=None, modelId=None, modelPredictionConfig=None, trainingStartedAt=None, trainingCompletedAt=None, featureGroupVersions=None, error=None, pendingDeploymentIds=None, failedDeploymentIds=None, cpuSize=None, memory=None, automlComplete=None, trainingFeatureGroupIds=None, trainingVectorStoreVersions=None, bestAlgorithm=None, defaultAlgorithm=None, featureAnalysisStatus=None, dataClusterInfo=None, customAlgorithmConfigs=None, trainedModelTypes=None, useGpu=None, partialComplete=None, modelFeatureGroupSchemaMappings=None, codeSource={}, modelConfig={}, deployableAlgorithms={}):
        super().__init__(client, modelVersion)
        self.model_version = modelVersion
        self.status = status
        self.model_id = modelId
        self.model_prediction_config = modelPredictionConfig
        self.training_started_at = trainingStartedAt
        self.training_completed_at = trainingCompletedAt
        self.feature_group_versions = featureGroupVersions
        self.error = error
        self.pending_deployment_ids = pendingDeploymentIds
        self.failed_deployment_ids = failedDeploymentIds
        self.cpu_size = cpuSize
        self.memory = memory
        self.automl_complete = automlComplete
        self.training_feature_group_ids = trainingFeatureGroupIds
        self.training_vector_store_versions = trainingVectorStoreVersions
        self.best_algorithm = bestAlgorithm
        self.default_algorithm = defaultAlgorithm
        self.feature_analysis_status = featureAnalysisStatus
        self.data_cluster_info = dataClusterInfo
        self.custom_algorithm_configs = customAlgorithmConfigs
        self.trained_model_types = trainedModelTypes
        self.use_gpu = useGpu
        self.partial_complete = partialComplete
        self.model_feature_group_schema_mappings = modelFeatureGroupSchemaMappings
        self.code_source = client._build_class(CodeSource, codeSource)
        self.model_config = client._build_class(TrainingConfig, modelConfig)
        self.deployable_algorithms = client._build_class(
            DeployableAlgorithm, deployableAlgorithms)

    def __repr__(self):
        repr_dict = {f'model_version': repr(self.model_version), f'status': repr(self.status), f'model_id': repr(self.model_id), f'model_prediction_config': repr(self.model_prediction_config), f'training_started_at': repr(self.training_started_at), f'training_completed_at': repr(self.training_completed_at), f'feature_group_versions': repr(self.feature_group_versions), f'error': repr(self.error), f'pending_deployment_ids': repr(self.pending_deployment_ids), f'failed_deployment_ids': repr(self.failed_deployment_ids), f'cpu_size': repr(self.cpu_size), f'memory': repr(self.memory), f'automl_complete': repr(self.automl_complete), f'training_feature_group_ids': repr(self.training_feature_group_ids), f'training_vector_store_versions': repr(
            self.training_vector_store_versions), f'best_algorithm': repr(self.best_algorithm), f'default_algorithm': repr(self.default_algorithm), f'feature_analysis_status': repr(self.feature_analysis_status), f'data_cluster_info': repr(self.data_cluster_info), f'custom_algorithm_configs': repr(self.custom_algorithm_configs), f'trained_model_types': repr(self.trained_model_types), f'use_gpu': repr(self.use_gpu), f'partial_complete': repr(self.partial_complete), f'model_feature_group_schema_mappings': repr(self.model_feature_group_schema_mappings), f'code_source': repr(self.code_source), f'model_config': repr(self.model_config), f'deployable_algorithms': repr(self.deployable_algorithms)}
        class_name = "ModelVersion"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'model_version': self.model_version, 'status': self.status, 'model_id': self.model_id, 'model_prediction_config': self.model_prediction_config, 'training_started_at': self.training_started_at, 'training_completed_at': self.training_completed_at, 'feature_group_versions': self.feature_group_versions, 'error': self.error, 'pending_deployment_ids': self.pending_deployment_ids, 'failed_deployment_ids': self.failed_deployment_ids, 'cpu_size': self.cpu_size, 'memory': self.memory, 'automl_complete': self.automl_complete, 'training_feature_group_ids': self.training_feature_group_ids, 'training_vector_store_versions': self.training_vector_store_versions,
                'best_algorithm': self.best_algorithm, 'default_algorithm': self.default_algorithm, 'feature_analysis_status': self.feature_analysis_status, 'data_cluster_info': self.data_cluster_info, 'custom_algorithm_configs': self.custom_algorithm_configs, 'trained_model_types': self.trained_model_types, 'use_gpu': self.use_gpu, 'partial_complete': self.partial_complete, 'model_feature_group_schema_mappings': self.model_feature_group_schema_mappings, 'code_source': self._get_attribute_as_dict(self.code_source), 'model_config': self._get_attribute_as_dict(self.model_config), 'deployable_algorithms': self._get_attribute_as_dict(self.deployable_algorithms)}
        return {key: value for key, value in resp.items() if value is not None}

    def describe_train_test_data_split_feature_group_version(self):
        """
        Get the train and test data split for a trained model by model version. This is only supported for models with custom algorithms.

        Args:
            model_version (str): The unique version ID of the model version.

        Returns:
            FeatureGroupVersion: The feature group version containing the training data and folds information.
        """
        return self.client.describe_train_test_data_split_feature_group_version(self.model_version)

    def set_model_objective(self, metric: str = None):
        """
        Sets the best model for all model instances of the model based on the specified metric, and updates the training configuration to use the specified metric for any future model versions.

        If metric is set to None, then just use the default selection


        Args:
            metric (str): The metric to use to determine the best model.
        """
        return self.client.set_model_objective(self.model_version, metric)

    def query_test_point_predictions(self, algorithm: str, to_row: int, from_row: int = 0, sql_where_clause: str = ''):
        """
        Query the test points predictions data for a specific algorithm.

        Args:
            algorithm (str): The algorithm id
            to_row (int): Ending row index to return.
            from_row (int): Starting row index to return.
            sql_where_clause (str): The SQL WHERE clause used to filter the data.

        Returns:
            TestPointPredictions: TestPointPrediction
        """
        return self.client.query_test_point_predictions(self.model_version, algorithm, to_row, from_row, sql_where_clause)

    def get_feature_group_schemas_for(self):
        """
        Gets the schema for all feature groups used in the model version.

        Args:
            model_version (str): Unique string identifier for the version of the model.

        Returns:
            list[ModelVersionFeatureGroupSchema]: List of schema for all feature groups used in the model version.
        """
        return self.client.get_feature_group_schemas_for_model_version(self.model_version)

    def delete(self):
        """
        Deletes the specified model version. Model versions which are currently used in deployments cannot be deleted.

        Args:
            model_version (str): The unique identifier of the model version to delete.
        """
        return self.client.delete_model_version(self.model_version)

    def export_model_artifact_as_feature_group(self, table_name: str, artifact_type: str):
        """
        Exports metric artifact data for a model as a feature group.

        Args:
            table_name (str): Name of the feature group table to create.
            artifact_type (str): EvalArtifact enum specifying which artifact to export.

        Returns:
            FeatureGroup: The created feature group.
        """
        return self.client.export_model_artifact_as_feature_group(self.model_version, table_name, artifact_type)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            ModelVersion: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Retrieves a full description of the specified model version.

        Args:
            model_version (str): Unique string identifier of the model version.

        Returns:
            ModelVersion: A model version.
        """
        return self.client.describe_model_version(self.model_version)

    def get_feature_importance_by(self):
        """
        Gets the feature importance calculated by various methods for the model.

        Args:
            model_version (str): Unique string identifier for the model version.

        Returns:
            FeatureImportance: Feature importances for the model.
        """
        return self.client.get_feature_importance_by_model_version(self.model_version)

    def get_training_data_logs(self):
        """
        Retrieves the data preparation logs during model training.

        Args:
            model_version (str): The unique version ID of the model version.

        Returns:
            list[DataPrepLogs]: A list of logs.
        """
        return self.client.get_training_data_logs(self.model_version)

    def get_training_logs(self, stdout: bool = False, stderr: bool = False):
        """
        Returns training logs for the model.

        Args:
            stdout (bool): Set True to get info logs.
            stderr (bool): Set True to get error logs.

        Returns:
            FunctionLogs: A function logs object.
        """
        return self.client.get_training_logs(self.model_version, stdout, stderr)

    def export_custom(self, output_location: str, algorithm: str = None):
        """
        Bundle custom model artifacts to a zip file, and export to the specified location.

        Args:
            output_location (str): Location to export the model artifacts results. For example, s3://a-bucket/
            algorithm (str): The algorithm to be exported. Optional if there's only one custom algorithm in the model version.

        Returns:
            ModelArtifactsExport: Object describing the export and its status.
        """
        return self.client.export_custom_model_version(self.model_version, output_location, algorithm)

    def wait_for_training(self, timeout=None):
        """
        A waiting call until model gets trained.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        start_time = time.time()
        while True:
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f'Maximum wait time of {timeout}s exceeded')
            self.refresh()
            if self.partial_complete:
                return self
            time.sleep(5)

    def wait_for_full_automl(self, timeout=None):
        """
        A waiting call until full AutoML cycle is completed.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        start_time = time.time()
        model_version = None
        while True:
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f'Maximum wait time of {timeout}s exceeded')
            model_version = self.client._call_api('describeModelVersion', 'GET', query_params={
                                                  'modelVersion': self.id, 'waitForFullAutoml': True}, parse_type=ModelVersion)
            if model_version.status not in {'PENDING', 'TRAINING'} and not model_version.pending_deployment_ids and model_version.feature_analysis_status not in {'ANALYZING', 'PENDING'}:
                break
            time.sleep(5)
        # not calling self.refresh() due to that doesn't accept waitForFullAutoml=True and result may be inconsistent
        self.__dict__.update(model_version.__dict__)
        return self

    def get_status(self):
        """
        Gets the status of the model version under training.

        Returns:
            str: A string describing the status of a model training (pending, complete, etc.).
        """
        return self.describe().status

    def get_train_test_feature_group_as_pandas(self):
        """
        Get the model train test data split feature group of the model version as pandas data frame.

        Returns:
            pandas.Dataframe: A pandas dataframe for the training data with fold column.
        """
        return self.client.describe_train_test_data_split_feature_group_version(self.model_version).load_as_pandas()
