from .return_class import AbstractApiClass


class TrainingConfigOptions(AbstractApiClass):
    """
        Training options for a model

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name of the parameter
            dataType (str): The type of input required for this option
            valueType (str): If the data_type is of type DICT_VALUES, this field specifies the expected value type of the values
            valueOptions (list of string): The list of valid values for DICT_VALUES
            value (string/number): The value of this option
            default (boolean/string/number): The default value for this option
            options (list of json objects): A list of options for this parameter
            description (str): A description of the parameter
            required (bool): True if the parameter is required for training
            lastModelValue (string/number): The last value used to train a model in this project
            needsRefresh (bool): 
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
        return f"TrainingConfigOptions(name={repr(self.name)},\n  data_type={repr(self.data_type)},\n  value_type={repr(self.value_type)},\n  value_options={repr(self.value_options)},\n  value={repr(self.value)},\n  default={repr(self.default)},\n  options={repr(self.options)},\n  description={repr(self.description)},\n  required={repr(self.required)},\n  last_model_value={repr(self.last_model_value)},\n  needs_refresh={repr(self.needs_refresh)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'data_type': self.data_type, 'value_type': self.value_type, 'value_options': self.value_options, 'value': self.value, 'default': self.default, 'options': self.options, 'description': self.description, 'required': self.required, 'last_model_value': self.last_model_value, 'needs_refresh': self.needs_refresh}
