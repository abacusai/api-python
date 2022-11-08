from .code_source import CodeSource
from .return_class import AbstractApiClass


class Algorithm(AbstractApiClass):
    """
        Customer created algorithm

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name of the algorithm
            problemType (str): The type of the problem this algorithm will work on
            externalProblemType (str): The problem type name shown to external user
            createdAt (str): 
            updatedAt (str): 
            isDefaultEnabled (bool): Whether train with the algorithm by default
            trainingInputMappings (dict): The mappings for train function parameters' names, e.g. names for training data, name for training config
            trainFunctionName (str): Name of the function found in the source code that will be executed to train the model. It is not executed when this function is run.
            predictFunctionName (str): Name of the function found in the source code that will be executed run predictions through model. It is not executed when this function is run.
            predictManyFunctionName (str): Name of the function found in the source code that will be executed for batch prediction of the model. It is not executed when this function is run.
            initializeFunctionName (str): Name of the function found in the source code to initialize the trained model before using it to make predictions using the model
            configOptions (dict): Map dataset types and configs to train function parameter names
            algorithm (str): The unique identifier of the algorithm
            useGpu (bool): 
            codeSource (CodeSource): 
    """

    def __init__(self, client, name=None, problemType=None, externalProblemType=None, createdAt=None, updatedAt=None, isDefaultEnabled=None, trainingInputMappings=None, trainFunctionName=None, predictFunctionName=None, predictManyFunctionName=None, initializeFunctionName=None, configOptions=None, algorithm=None, useGpu=None, codeSource={}):
        super().__init__(client, None)
        self.name = name
        self.problem_type = problemType
        self.external_problem_type = externalProblemType
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.is_default_enabled = isDefaultEnabled
        self.training_input_mappings = trainingInputMappings
        self.train_function_name = trainFunctionName
        self.predict_function_name = predictFunctionName
        self.predict_many_function_name = predictManyFunctionName
        self.initialize_function_name = initializeFunctionName
        self.config_options = configOptions
        self.algorithm = algorithm
        self.use_gpu = useGpu
        self.code_source = client._build_class(CodeSource, codeSource)

    def __repr__(self):
        return f"Algorithm(name={repr(self.name)},\n  problem_type={repr(self.problem_type)},\n  external_problem_type={repr(self.external_problem_type)},\n  created_at={repr(self.created_at)},\n  updated_at={repr(self.updated_at)},\n  is_default_enabled={repr(self.is_default_enabled)},\n  training_input_mappings={repr(self.training_input_mappings)},\n  train_function_name={repr(self.train_function_name)},\n  predict_function_name={repr(self.predict_function_name)},\n  predict_many_function_name={repr(self.predict_many_function_name)},\n  initialize_function_name={repr(self.initialize_function_name)},\n  config_options={repr(self.config_options)},\n  algorithm={repr(self.algorithm)},\n  use_gpu={repr(self.use_gpu)},\n  code_source={repr(self.code_source)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'problem_type': self.problem_type, 'external_problem_type': self.external_problem_type, 'created_at': self.created_at, 'updated_at': self.updated_at, 'is_default_enabled': self.is_default_enabled, 'training_input_mappings': self.training_input_mappings, 'train_function_name': self.train_function_name, 'predict_function_name': self.predict_function_name, 'predict_many_function_name': self.predict_many_function_name, 'initialize_function_name': self.initialize_function_name, 'config_options': self.config_options, 'algorithm': self.algorithm, 'use_gpu': self.use_gpu, 'code_source': self._get_attribute_as_dict(self.code_source)}
