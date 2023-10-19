from .return_class import AbstractApiClass


class TrainingConfigOptions(AbstractApiClass):
    """
        Training options for a model

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name of the parameter
            dataType (str): The type of input required for this option
            valueType (str): If the data_type is of type DICT_VALUES, this field specifies the expected value type of the values
            valueOptions (list[str]): The list of valid values for DICT_VALUES
            value (optional[str, int, float, bool]): The value of this option
            default (optional[str, int, float, bool]): The default value for this option
            options (list[dict]): A list of options for this parameter
            description (str): A description of the parameter
            required (bool): True if the parameter is required for training
            lastModelValue (optional[str, int, float, bool]): The last value used to train a model in this project
            needsRefresh (bool): True if training config needs to be fetched again when this config option is changed
    """

    def __init__(self, client, name=None, dataType=None, valueType=None, valueOptions=None, value=None, default=None, options=None, description=None, required=None, lastModelValue=None, needsRefresh=None):
        super().__init__(client, None)
        self.name = name
        self.data_type = dataType
        self.value_type = valueType
        self.value_options = valueOptions
        self.value = value
        self.default = default
        self.options = options
        self.description = description
        self.required = required
        self.last_model_value = lastModelValue
        self.needs_refresh = needsRefresh

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'data_type': repr(self.data_type), f'value_type': repr(self.value_type), f'value_options': repr(self.value_options), f'value': repr(self.value), f'default': repr(
            self.default), f'options': repr(self.options), f'description': repr(self.description), f'required': repr(self.required), f'last_model_value': repr(self.last_model_value), f'needs_refresh': repr(self.needs_refresh)}
        class_name = "TrainingConfigOptions"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'data_type': self.data_type, 'value_type': self.value_type, 'value_options': self.value_options, 'value': self.value, 'default': self.default,
                'options': self.options, 'description': self.description, 'required': self.required, 'last_model_value': self.last_model_value, 'needs_refresh': self.needs_refresh}
        return {key: value for key, value in resp.items() if value is not None}
