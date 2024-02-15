from .code_source import CodeSource
from .return_class import AbstractApiClass


class Algorithm(AbstractApiClass):
    """
        Customer created algorithm

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name of the algorithm
            problemType (str): The type of the problem this algorithm will work on
            createdAt (str): When the algorithm was created
            updatedAt (str): When the algorithm was last updated
            isDefaultEnabled (bool): Whether train with the algorithm by default
            trainingInputMappings (dict): The mappings for train function parameters' names, e.g. names for training data, name for training config
            trainFunctionName (str): Name of the function found in the source code that will be executed to train the model. It is not executed when this function is run.
            predictFunctionName (str): Name of the function found in the source code that will be executed run predictions through model. It is not executed when this function is run.
            predictManyFunctionName (str): Name of the function found in the source code that will be executed for batch prediction of the model. It is not executed when this function is run.
            initializeFunctionName (str): Name of the function found in the source code to initialize the trained model before using it to make predictions using the model
            configOptions (dict): Map dataset types and configs to train function parameter names
            algorithmId (str): The unique identifier of the algorithm
            useGpu (bool): Whether to use gpu for model training
            algorithmTrainingConfig (dict): The algorithm specific training config
            onlyOfflineDeployable (bool): Whether or not the algorithm is only allowed to be deployed offline
            codeSource (CodeSource): Info about the source code of the algorithm
    """

    def __init__(self, client, name=None, problemType=None, createdAt=None, updatedAt=None, isDefaultEnabled=None, trainingInputMappings=None, trainFunctionName=None, predictFunctionName=None, predictManyFunctionName=None, initializeFunctionName=None, configOptions=None, algorithmId=None, useGpu=None, algorithmTrainingConfig=None, onlyOfflineDeployable=None, codeSource={}):
        super().__init__(client, algorithmId)
        self.name = name
        self.problem_type = problemType
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.is_default_enabled = isDefaultEnabled
        self.training_input_mappings = trainingInputMappings
        self.train_function_name = trainFunctionName
        self.predict_function_name = predictFunctionName
        self.predict_many_function_name = predictManyFunctionName
        self.initialize_function_name = initializeFunctionName
        self.config_options = configOptions
        self.algorithm_id = algorithmId
        self.use_gpu = useGpu
        self.algorithm_training_config = algorithmTrainingConfig
        self.only_offline_deployable = onlyOfflineDeployable
        self.code_source = client._build_class(CodeSource, codeSource)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'problem_type': repr(self.problem_type), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at), f'is_default_enabled': repr(self.is_default_enabled), f'training_input_mappings': repr(self.training_input_mappings), f'train_function_name': repr(self.train_function_name), f'predict_function_name': repr(self.predict_function_name), f'predict_many_function_name': repr(
            self.predict_many_function_name), f'initialize_function_name': repr(self.initialize_function_name), f'config_options': repr(self.config_options), f'algorithm_id': repr(self.algorithm_id), f'use_gpu': repr(self.use_gpu), f'algorithm_training_config': repr(self.algorithm_training_config), f'only_offline_deployable': repr(self.only_offline_deployable), f'code_source': repr(self.code_source)}
        class_name = "Algorithm"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'problem_type': self.problem_type, 'created_at': self.created_at, 'updated_at': self.updated_at, 'is_default_enabled': self.is_default_enabled, 'training_input_mappings': self.training_input_mappings, 'train_function_name': self.train_function_name, 'predict_function_name': self.predict_function_name, 'predict_many_function_name':
                self.predict_many_function_name, 'initialize_function_name': self.initialize_function_name, 'config_options': self.config_options, 'algorithm_id': self.algorithm_id, 'use_gpu': self.use_gpu, 'algorithm_training_config': self.algorithm_training_config, 'only_offline_deployable': self.only_offline_deployable, 'code_source': self._get_attribute_as_dict(self.code_source)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
