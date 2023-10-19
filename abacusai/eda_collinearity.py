from .return_class import AbstractApiClass


class EdaCollinearity(AbstractApiClass):
    """
        Eda Collinearity of the latest version of the data between all the features.

        Args:
            client (ApiClient): An authenticated API Client instance
            columnNames (list): Name of all the features in the y axis of the collinearity matrix
            collinearityMatrix (dict): A dict describing the collinearity between all the features
            groupFeatureDict (dict): A dict describing the index of the group from collinearity_groups a feature exists in
            collinearityGroups (list): Groups created based on a collinearity threshold of 0.7
            columnNamesX (list): Name of all the features in the x axis of the collinearity matrix
    """

    def __init__(self, client, columnNames=None, collinearityMatrix=None, groupFeatureDict=None, collinearityGroups=None, columnNamesX=None):
        super().__init__(client, None)
        self.column_names = columnNames
        self.collinearity_matrix = collinearityMatrix
        self.group_feature_dict = groupFeatureDict
        self.collinearity_groups = collinearityGroups
        self.column_names_x = columnNamesX

    def __repr__(self):
        repr_dict = {f'column_names': repr(self.column_names), f'collinearity_matrix': repr(self.collinearity_matrix), f'group_feature_dict': repr(
            self.group_feature_dict), f'collinearity_groups': repr(self.collinearity_groups), f'column_names_x': repr(self.column_names_x)}
        class_name = "EdaCollinearity"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'column_names': self.column_names, 'collinearity_matrix': self.collinearity_matrix, 'group_feature_dict':
                self.group_feature_dict, 'collinearity_groups': self.collinearity_groups, 'column_names_x': self.column_names_x}
        return {key: value for key, value in resp.items() if value is not None}
