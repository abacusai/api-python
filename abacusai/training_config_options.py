

class TrainingConfigOptions():
    '''

    '''

    def __init__(self, client, name=None, dataType=None, value=None, default=None, options=None, description=None, required=None, lastModelValue=None):
        self.client = client
        self.id = None
        self.name = name
        self.data_type = dataType
        self.value = value
        self.default = default
        self.options = options
        self.description = description
        self.required = required
        self.last_model_value = lastModelValue

    def __repr__(self):
        return f"TrainingConfigOptions(name={repr(self.name)}, data_type={repr(self.data_type)}, value={repr(self.value)}, default={repr(self.default)}, options={repr(self.options)}, description={repr(self.description)}, required={repr(self.required)}, last_model_value={repr(self.last_model_value)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'name': self.name, 'data_type': self.data_type, 'value': self.value, 'default': self.default, 'options': self.options, 'description': self.description, 'required': self.required, 'last_model_value': self.last_model_value}
