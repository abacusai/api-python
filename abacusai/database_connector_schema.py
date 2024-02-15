from .database_connector_column import DatabaseConnectorColumn
from .return_class import AbstractApiClass


class DatabaseConnectorSchema(AbstractApiClass):
    """
        A schema description for a table from a database connector

        Args:
            client (ApiClient): An authenticated API Client instance
            tableName (str): The unique name of the table.
            columns (DatabaseConnectorColumn): List of columns in the table.
    """

    def __init__(self, client, tableName=None, columns={}):
        super().__init__(client, None)
        self.table_name = tableName
        self.columns = client._build_class(DatabaseConnectorColumn, columns)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'table_name': repr(
            self.table_name), f'columns': repr(self.columns)}
        class_name = "DatabaseConnectorSchema"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'table_name': self.table_name,
                'columns': self._get_attribute_as_dict(self.columns)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
