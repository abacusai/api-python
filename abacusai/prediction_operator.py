from .code_source import CodeSource
from .prediction_operator_version import PredictionOperatorVersion
from .refresh_schedule import RefreshSchedule
from .return_class import AbstractApiClass


class PredictionOperator(AbstractApiClass):
    """
        A prediction operator.

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name for the prediction operator.
            predictionOperatorId (str): The unique identifier of the prediction operator.
            createdAt (str): Date and time at which the prediction operator was created.
            updatedAt (str): Date and time at which the prediction operator was updated.
            projectId (str): The project this prediction operator belongs to.
            predictFunctionName (str): Name of the function found in the source code that will be executed to run predictions.
            sourceCode (str): Python code used to make the prediction operator.
            initializeFunctionName (str): Name of the optional initialize function found in the source code. This function will generate anything used by predictions, based on input feature groups.
            notebookId (str): The unique string identifier of the notebook used to create or edit the prediction operator.
            memory (int): Memory in GB specified for the prediction operator.
            useGpu (bool): Whether this prediction operator is using gpu.
            featureGroupIds (list): A list of Feature Group IDs used for initializing.
            featureGroupTableNames (list): A list of Feature Group table names used for initializing.
            codeSource (CodeSource): If a python model, information on the source code.
            latestPredictionOperatorVersion (PredictionOperatorVersion): The unique string identifier of the latest version.
            refreshSchedules (RefreshSchedule): List of refresh schedules that indicate when the next prediction operator version will be processed
    """

    def __init__(self, client, name=None, predictionOperatorId=None, createdAt=None, updatedAt=None, projectId=None, predictFunctionName=None, sourceCode=None, initializeFunctionName=None, notebookId=None, memory=None, useGpu=None, featureGroupIds=None, featureGroupTableNames=None, codeSource={}, refreshSchedules={}, latestPredictionOperatorVersion={}):
        super().__init__(client, predictionOperatorId)
        self.name = name
        self.prediction_operator_id = predictionOperatorId
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.project_id = projectId
        self.predict_function_name = predictFunctionName
        self.source_code = sourceCode
        self.initialize_function_name = initializeFunctionName
        self.notebook_id = notebookId
        self.memory = memory
        self.use_gpu = useGpu
        self.feature_group_ids = featureGroupIds
        self.feature_group_table_names = featureGroupTableNames
        self.code_source = client._build_class(CodeSource, codeSource)
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)
        self.latest_prediction_operator_version = client._build_class(
            PredictionOperatorVersion, latestPredictionOperatorVersion)

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'prediction_operator_id': repr(self.prediction_operator_id), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at), f'project_id': repr(self.project_id), f'predict_function_name': repr(self.predict_function_name), f'source_code': repr(self.source_code), f'initialize_function_name': repr(self.initialize_function_name), f'notebook_id': repr(
            self.notebook_id), f'memory': repr(self.memory), f'use_gpu': repr(self.use_gpu), f'feature_group_ids': repr(self.feature_group_ids), f'feature_group_table_names': repr(self.feature_group_table_names), f'code_source': repr(self.code_source), f'refresh_schedules': repr(self.refresh_schedules), f'latest_prediction_operator_version': repr(self.latest_prediction_operator_version)}
        class_name = "PredictionOperator"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'prediction_operator_id': self.prediction_operator_id, 'created_at': self.created_at, 'updated_at': self.updated_at, 'project_id': self.project_id, 'predict_function_name': self.predict_function_name, 'source_code': self.source_code, 'initialize_function_name': self.initialize_function_name, 'notebook_id': self.notebook_id, 'memory': self.memory,
                'use_gpu': self.use_gpu, 'feature_group_ids': self.feature_group_ids, 'feature_group_table_names': self.feature_group_table_names, 'code_source': self._get_attribute_as_dict(self.code_source), 'refresh_schedules': self._get_attribute_as_dict(self.refresh_schedules), 'latest_prediction_operator_version': self._get_attribute_as_dict(self.latest_prediction_operator_version)}
        return {key: value for key, value in resp.items() if value is not None}

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            PredictionOperator: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Describe an existing prediction operator.

        Args:
            prediction_operator_id (str): The unique ID of the prediction operator.

        Returns:
            PredictionOperator: The requested prediction operator object.
        """
        return self.client.describe_prediction_operator(self.prediction_operator_id)

    def update(self, name: str = None, feature_group_ids: list = None, source_code: str = None, initialize_function_name: str = None, predict_function_name: str = None, cpu_size: str = None, memory: int = None, package_requirements: list = None, use_gpu: bool = None):
        """
        Update an existing prediction operator.

        Args:
            name (str): Name of the prediction operator.
            feature_group_ids (list): List of feature groups that are supplied to the initialize function as parameters. Each of the parameters are materialized Dataframes. The order should match the initialize function's parameters.
            source_code (str): Contents of a valid Python source code file. The source code should contain the function `predictFunctionName`, and the function 'initializeFunctionName' if defined.
            initialize_function_name (str): Name of the optional initialize function found in the source code. This function will generate anything used by predictions, based on input feature groups.
            predict_function_name (str): Name of the function found in the source code that will be executed to run predictions.
            cpu_size (str): Size of the CPU for the prediction operator.
            memory (int): Memory (in GB) for the  prediction operator.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            use_gpu (bool): Whether this prediction operator needs gpu.

        Returns:
            PredictionOperator: The updated prediction operator object.
        """
        return self.client.update_prediction_operator(self.prediction_operator_id, name, feature_group_ids, source_code, initialize_function_name, predict_function_name, cpu_size, memory, package_requirements, use_gpu)

    def delete(self):
        """
        Delete an existing prediction operator.

        Args:
            prediction_operator_id (str): The unique ID of the prediction operator.
        """
        return self.client.delete_prediction_operator(self.prediction_operator_id)

    def deploy(self, auto_deploy: bool = True):
        """
        Deploy the prediction operator.

        Args:
            auto_deploy (bool): Flag to enable the automatic deployment when a new prediction operator version is created.

        Returns:
            Deployment: The created deployment object.
        """
        return self.client.deploy_prediction_operator(self.prediction_operator_id, auto_deploy)

    def create_version(self):
        """
        Create a new version of the prediction operator.

        Args:
            prediction_operator_id (str): The unique ID of the prediction operator.

        Returns:
            PredictionOperatorVersion: The created prediction operator version object.
        """
        return self.client.create_prediction_operator_version(self.prediction_operator_id)

    def list_versions(self):
        """
        List all the prediction operator versions for a prediction operator.

        Args:
            prediction_operator_id (str): The unique ID of the prediction operator.

        Returns:
            list[PredictionOperatorVersion]: A list of prediction operator version objects.
        """
        return self.client.list_prediction_operator_versions(self.prediction_operator_id)
