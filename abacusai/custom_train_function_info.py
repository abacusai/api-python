from .return_class import AbstractApiClass


class CustomTrainFunctionInfo(AbstractApiClass):
    """
        Information about how to call the customer provided train function.

        Args:
            client (ApiClient): An authenticated API Client instance
            trainingDataParameterNameMapping (dict): The mapping from feature group type to the dataframe parameter name
            schemaMappings (dict): The feature type to feature name mapping for each dataframe
            trainDataParameterToFeatureGroupIds (dict): The mapping from the dataframe parameter name to the feature group id backing the data
            trainingConfig (dict): The configs for training
    """

    def __init__(self, client, trainingDataParameterNameMapping=None, schemaMappings=None, trainDataParameterToFeatureGroupIds=None, trainingConfig=None):
        super().__init__(client, None)
        self.training_data_parameter_name_mapping = trainingDataParameterNameMapping
        self.schema_mappings = schemaMappings
        self.train_data_parameter_to_feature_group_ids = trainDataParameterToFeatureGroupIds
        self.training_config = trainingConfig

    def __repr__(self):
        return f"CustomTrainFunctionInfo(training_data_parameter_name_mapping={repr(self.training_data_parameter_name_mapping)},\n  schema_mappings={repr(self.schema_mappings)},\n  train_data_parameter_to_feature_group_ids={repr(self.train_data_parameter_to_feature_group_ids)},\n  training_config={repr(self.training_config)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'training_data_parameter_name_mapping': self.training_data_parameter_name_mapping, 'schema_mappings': self.schema_mappings, 'train_data_parameter_to_feature_group_ids': self.train_data_parameter_to_feature_group_ids, 'training_config': self.training_config}
