import time

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
            modelConfig (dict): The training config options used to train this model.
            modelPredictionConfig (dict): The prediction config options for the model.
            trainingStartedAt (str): The start time and date of the training process.
            trainingCompletedAt (str): The end time and date of the training process.
            datasetVersions (list of unique string identifiers): Comma separated list of Dataset version IDs that this refresh pipeline run is monitoring.
            featureGroupVersions (list): 
            error (str): Relevant error if the status is FAILED.
            pendingDeploymentIds (list): List of deployment IDs where deployment is pending.
            failedDeploymentIds (list): List of failed deployment IDs.
            cpuSize (str): Cpu size specified for the python model training.
            memory (int): Memory in GB specified for the python model training.
            automlComplete (bool): If true, all algorithms have compelted training
            trainingFeatureGroupIds (list of unique string identifiers): The unique identifiers of the feature group used as the inputs during training to create this ModelVersion.
            deployableAlgorithms (dict): List of deployable algorithms
            bestAlgorithm (dict): Best performing algorithm
            defaultAlgorithm (dict): Default algorithm that the user has selected
            featureAnalysisStatus (str): lifecycle of the feature analysis stage
            dataClusterInfo (dict): Information about the models for different data clusters
            codeSource (CodeSource): If a python model, information on where the source code
    """

    def __init__(self, client, modelVersion=None, status=None, modelId=None, modelConfig=None, modelPredictionConfig=None, trainingStartedAt=None, trainingCompletedAt=None, datasetVersions=None, featureGroupVersions=None, error=None, pendingDeploymentIds=None, failedDeploymentIds=None, cpuSize=None, memory=None, automlComplete=None, trainingFeatureGroupIds=None, deployableAlgorithms=None, bestAlgorithm=None, defaultAlgorithm=None, featureAnalysisStatus=None, dataClusterInfo=None, codeSource={}):
        super().__init__(client, modelVersion)
        self.model_version = modelVersion
        self.status = status
        self.model_id = modelId
        self.model_config = modelConfig
        self.model_prediction_config = modelPredictionConfig
        self.training_started_at = trainingStartedAt
        self.training_completed_at = trainingCompletedAt
        self.dataset_versions = datasetVersions
        self.feature_group_versions = featureGroupVersions
        self.error = error
        self.pending_deployment_ids = pendingDeploymentIds
        self.failed_deployment_ids = failedDeploymentIds
        self.cpu_size = cpuSize
        self.memory = memory
        self.automl_complete = automlComplete
        self.training_feature_group_ids = trainingFeatureGroupIds
        self.deployable_algorithms = deployableAlgorithms
        self.best_algorithm = bestAlgorithm
        self.default_algorithm = defaultAlgorithm
        self.feature_analysis_status = featureAnalysisStatus
        self.data_cluster_info = dataClusterInfo
        self.code_source = client._build_class(CodeSource, codeSource)

    def __repr__(self):
        return f"ModelVersion(model_version={repr(self.model_version)},\n  status={repr(self.status)},\n  model_id={repr(self.model_id)},\n  model_config={repr(self.model_config)},\n  model_prediction_config={repr(self.model_prediction_config)},\n  training_started_at={repr(self.training_started_at)},\n  training_completed_at={repr(self.training_completed_at)},\n  dataset_versions={repr(self.dataset_versions)},\n  feature_group_versions={repr(self.feature_group_versions)},\n  error={repr(self.error)},\n  pending_deployment_ids={repr(self.pending_deployment_ids)},\n  failed_deployment_ids={repr(self.failed_deployment_ids)},\n  cpu_size={repr(self.cpu_size)},\n  memory={repr(self.memory)},\n  automl_complete={repr(self.automl_complete)},\n  training_feature_group_ids={repr(self.training_feature_group_ids)},\n  deployable_algorithms={repr(self.deployable_algorithms)},\n  best_algorithm={repr(self.best_algorithm)},\n  default_algorithm={repr(self.default_algorithm)},\n  feature_analysis_status={repr(self.feature_analysis_status)},\n  data_cluster_info={repr(self.data_cluster_info)},\n  code_source={repr(self.code_source)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'model_version': self.model_version, 'status': self.status, 'model_id': self.model_id, 'model_config': self.model_config, 'model_prediction_config': self.model_prediction_config, 'training_started_at': self.training_started_at, 'training_completed_at': self.training_completed_at, 'dataset_versions': self.dataset_versions, 'feature_group_versions': self.feature_group_versions, 'error': self.error, 'pending_deployment_ids': self.pending_deployment_ids, 'failed_deployment_ids': self.failed_deployment_ids, 'cpu_size': self.cpu_size, 'memory': self.memory, 'automl_complete': self.automl_complete, 'training_feature_group_ids': self.training_feature_group_ids, 'deployable_algorithms': self.deployable_algorithms, 'best_algorithm': self.best_algorithm, 'default_algorithm': self.default_algorithm, 'feature_analysis_status': self.feature_analysis_status, 'data_cluster_info': self.data_cluster_info, 'code_source': self._get_attribute_as_dict(self.code_source)}

    def describe_train_test_data_split_feature_group_version(self):
        """
        Get the train and test data split for a trained model by model_version. Only supported for models with custom algorithms.

        Args:
            model_version (str): The unique version ID of the model version

        Returns:
            FeatureGroupVersion: The feature group version containing the training data and folds information.
        """
        return self.client.describe_train_test_data_split_feature_group_version(self.model_version)

    def delete(self):
        """
        Deletes the specified model version. Model Versions which are currently used in deployments cannot be deleted.

        Args:
            model_version (str): The ID of the model version to delete.
        """
        return self.client.delete_model_version(self.model_version)

    def export_model_artifact_as_feature_group(self, table_name: str, artifact_type: str):
        """
        Exports metric artifact data for a model as a feature group.

        Args:
            table_name (str): The name of the feature group table to create.
            artifact_type (str): An EvalArtifact enum of which artifact to export.

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
        Retrieves a full description of the specified model version

        Args:
            model_version (str): The unique version ID of the model version

        Returns:
            ModelVersion: A model version.
        """
        return self.client.describe_model_version(self.model_version)

    def get_training_data_logs(self):
        """
        Retrieves the data preparation logs during model training.

        Args:
            model_version (str): The unique version ID of the model version

        Returns:
            DataPrepLogs: A list of logs.
        """
        return self.client.get_training_data_logs(self.model_version)

    def get_training_logs(self, stdout: bool = False, stderr: bool = False):
        """
        Returns training logs for the model.

        Args:
            stdout (bool):  Set True to get info logs
            stderr (bool):  Set True to get error logs

        Returns:
            FunctionLogs: A function logs.
        """
        return self.client.get_training_logs(self.model_version, stdout, stderr)

    def ignore_lofo_features(self, threshold: float = None, top_n: int = 0):
        """


        Args:
            threshold (float): 
            top_n (int): 
        """
        return self.client.ignore_lofo_features(self.model_version, threshold, top_n)

    def wait_for_training(self, timeout=None):
        """
        A waiting call until model gets trained.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'TRAINING'}, delay=30, timeout=timeout)

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
            time.sleep(30)
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
