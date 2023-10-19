from .return_class import AbstractApiClass


class NestedFeature(AbstractApiClass):
    """
        A nested feature in a feature group

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The unique name of the column
            selectClause (str): The sql logic for creating this feature's data
            featureType (str): Feature Type of the Feature
            featureMapping (str): The Feature Mapping of the feature
            dataType (str): Data Type of the Feature
            dataUse (str): The data use of the feature
            sourceTable (str): The source table of the column
            originalName (str): The original name of the column
    """

    def __init__(self, client, name=None, selectClause=None, featureType=None, featureMapping=None, dataType=None, dataUse=None, sourceTable=None, originalName=None):
        super().__init__(client, None)
        self.name = name
        self.select_clause = selectClause
        self.feature_type = featureType
        self.feature_mapping = featureMapping
        self.data_type = dataType
        self.data_use = dataUse
        self.source_table = sourceTable
        self.original_name = originalName

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'select_clause': repr(self.select_clause), f'feature_type': repr(self.feature_type), f'feature_mapping': repr(
            self.feature_mapping), f'data_type': repr(self.data_type), f'data_use': repr(self.data_use), f'source_table': repr(self.source_table), f'original_name': repr(self.original_name)}
        class_name = "NestedFeature"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'select_clause': self.select_clause, 'feature_type': self.feature_type, 'feature_mapping': self.feature_mapping,
                'data_type': self.data_type, 'data_use': self.data_use, 'source_table': self.source_table, 'original_name': self.original_name}
        return {key: value for key, value in resp.items() if value is not None}
