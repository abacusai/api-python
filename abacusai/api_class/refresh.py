import dataclasses
from typing import Dict, List

from . import enums
from .abstract import ApiClass, _ApiClassFactory


@dataclasses.dataclass
class FeatureGroupExportConfig(ApiClass):
    """
    An abstract class for feature group exports.
    """
    connector_type: enums.ConnectorType = dataclasses.field(default=None, repr=False, init=False)

    @classmethod
    def _get_builder(cls):
        return _FeatureGroupExportConfigFactory


@dataclasses.dataclass
class FileConnectorExportConfig(FeatureGroupExportConfig):
    """
    File connector export config for feature groups

    Args:
        location (str): The location to export the feature group to
        export_file_format (str): The file format to export the feature group to
    """
    location: str = dataclasses.field(default=None)
    export_file_format: str = dataclasses.field(default=None)

    def __post_init__(self):
        self.connector_type = enums.ConnectorType.FILE

    def to_dict(self):
        return {
            'connector_type': self.connector_type,
            'location': self.location,
            'export_file_format': self.export_file_format,
        }


@dataclasses.dataclass
class DatabaseConnectorExportConfig(FeatureGroupExportConfig):
    """
    Database connector export config for feature groups

    Args:
        database_connector_id (str): The ID of the database connector to export the feature group to
        mode (str): The mode to export the feature group in
        object_name (str): The name of the object to export the feature group to
        id_column (str): The name of the ID column
        additional_id_columns (List[str]): Additional ID columns
        data_columns (Dict[str, str]): The data columns to export the feature group to
    """
    database_connector_id: str = dataclasses.field(default=None)
    mode: str = dataclasses.field(default=None)
    object_name: str = dataclasses.field(default=None)
    id_column: str = dataclasses.field(default=None)
    additional_id_columns: List[str] = dataclasses.field(default=None)
    data_columns: Dict[str, str] = dataclasses.field(default=None)

    def __post_init__(self):
        self.connector_type = enums.ConnectorType.DATABASE

    def to_dict(self):
        return {
            'connector_type': self.connector_type,
            'database_connector_id': self.database_connector_id,
            'mode': self.mode,
            'object_name': self.object_name,
            'id_column': self.id_column,
            'additional_id_columns': self.additional_id_columns,
            'data_columns': self.data_columns,
        }


@dataclasses.dataclass
class _FeatureGroupExportConfigFactory(_ApiClassFactory):
    config_abstract_class = FeatureGroupExportConfig
    config_class_key = 'connector_type'
    config_class_map = {
        enums.ConnectorType.FILE: FileConnectorExportConfig,
        enums.ConnectorType.DATABASE: DatabaseConnectorExportConfig,
    }
