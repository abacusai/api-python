from .return_class import AbstractApiClass


class Schema(AbstractApiClass):
    """
        A schema description for a feature

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The unique name of the feature.
            featureMapping (str): The mapping of the feature. The possible values will be based on the project's use-case. See the (Use Case Documentation)[https://api.abacus.ai/app/help/useCases] for more details.
            featureType (str): The underlying data type of each feature:  CATEGORICAL,  CATEGORICAL_LIST,  NUMERICAL,  TIMESTAMP,  TEXT,  EMAIL,  LABEL_LIST,  JSON,  OBJECT_REFERENCE.  Refer to the (guide on data types)[https://api.abacus.ai/app/help/class/FeatureType] for more information.
            dataType (str): The underlying data type of each feature:  INTEGER,  FLOAT,  STRING,  DATE,  DATETIME,  BOOLEAN,  LIST,  STRUCT.  Refer to the (guide on data types)[https://api.abacus.ai/app/help/class/DataType] for more information.
    """

    def __init__(self, client, name=None, featureMapping=None, featureType=None, dataType=None):
        super().__init__(client, None)
        self.name = name
        self.feature_mapping = featureMapping
        self.feature_type = featureType
        self.data_type = dataType

    def __repr__(self):
        return f"Schema(name={repr(self.name)},\n  feature_mapping={repr(self.feature_mapping)},\n  feature_type={repr(self.feature_type)},\n  data_type={repr(self.data_type)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'feature_mapping': self.feature_mapping, 'feature_type': self.feature_type, 'data_type': self.data_type}
