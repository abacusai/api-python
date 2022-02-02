from .return_class import AbstractApiClass


class DatasetColumn(AbstractApiClass):
    """
        A schema description for a column

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The unique name of the column.
            dataType (str): The underlying data type of each column:  INTEGER,  FLOAT,  STRING,  DATE,  DATETIME,  BOOLEAN,  LIST,  STRUCT.  Refer to the (guide on data types)[https://api.abacus.ai/app/help/class/DataType] for more information.
            featureType (str): Feature Type of the Feature
            originalName (str): The original name of the column
    """

    def __init__(self, client, name=None, dataType=None, featureType=None, originalName=None):
        super().__init__(client, None)
        self.name = name
        self.data_type = dataType
        self.feature_type = featureType
        self.original_name = originalName

    def __repr__(self):
        return f"DatasetColumn(name={repr(self.name)},\n  data_type={repr(self.data_type)},\n  feature_type={repr(self.feature_type)},\n  original_name={repr(self.original_name)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'data_type': self.data_type, 'feature_type': self.feature_type, 'original_name': self.original_name}
