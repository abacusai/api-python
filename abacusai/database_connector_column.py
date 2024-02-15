from .return_class import AbstractApiClass


class DatabaseConnectorColumn(AbstractApiClass):
    """
        A schema description for a column from a database connector

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The unique name of the column.
            externalDataType (str): The data type of column in the external database system.
    """

    def __init__(self, client, name=None, externalDataType=None):
        super().__init__(client, None)
        self.name = name
        self.external_data_type = externalDataType
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'external_data_type': repr(
            self.external_data_type)}
        class_name = "DatabaseConnectorColumn"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'external_data_type': self.external_data_type}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
