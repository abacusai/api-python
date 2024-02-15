from .return_class import AbstractApiClass


class PointInTimeGroupFeature(AbstractApiClass):
    """
        A point in time group feature

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name of the feature
            expression (str): SQL Aggregate expression which can convert a sequence of rows into a scalar value.
            pitOperationType (str): The operation used in point in time feature generation
            pitOperationConfig (dict): The configuration used as input to the operation type
    """

    def __init__(self, client, name=None, expression=None, pitOperationType=None, pitOperationConfig=None):
        super().__init__(client, None)
        self.name = name
        self.expression = expression
        self.pit_operation_type = pitOperationType
        self.pit_operation_config = pitOperationConfig
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'expression': repr(self.expression), f'pit_operation_type': repr(
            self.pit_operation_type), f'pit_operation_config': repr(self.pit_operation_config)}
        class_name = "PointInTimeGroupFeature"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'expression': self.expression, 'pit_operation_type':
                self.pit_operation_type, 'pit_operation_config': self.pit_operation_config}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
