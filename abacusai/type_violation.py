from .return_class import AbstractApiClass


class TypeViolation(AbstractApiClass):
    """
        Summary of important type mismatches for a feature discovered by a model monitoring instance

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): Name of feature.
            trainingDataType (str): Data type of training feature that doesn't match the type of the corresponding prediction feature.
            predictionDataType (str): Data type of prediction feature that doesn't match the type of the corresponding training feature.
    """

    def __init__(self, client, name=None, trainingDataType=None, predictionDataType=None):
        super().__init__(client, None)
        self.name = name
        self.training_data_type = trainingDataType
        self.prediction_data_type = predictionDataType

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'training_data_type': repr(
            self.training_data_type), f'prediction_data_type': repr(self.prediction_data_type)}
        class_name = "TypeViolation"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'training_data_type': self.training_data_type,
                'prediction_data_type': self.prediction_data_type}
        return {key: value for key, value in resp.items() if value is not None}
