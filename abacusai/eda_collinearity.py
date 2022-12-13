from .return_class import AbstractApiClass


class EdaCollinearity(AbstractApiClass):
    """
        Eda Collinearity of the latest version of the data between all the features.

        Args:
            client (ApiClient): An authenticated API Client instance
            columnNames (list): Name of all the features in the data
            collinearityMatrix (CollinearityRecord): A CollinearityRecord describing the collinearity between all the features
    """

    def __init__(self, client, columnNames=None, collinearityMatrix={}):
        super().__init__(client, None)
        self.column_names = columnNames
        self.collinearity_matrix = client._build_class(
            CollinearityRecord, collinearityMatrix)

    def __repr__(self):
        return f"EdaCollinearity(column_names={repr(self.column_names)},\n  collinearity_matrix={repr(self.collinearity_matrix)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'column_names': self.column_names, 'collinearity_matrix': self._get_attribute_as_dict(self.collinearity_matrix)}
