import dataclasses
from typing import Dict, List

from . import enums
from .abstract import ApiClass, _ApiClassFactory


@dataclasses.dataclass
class FeatureGroupExportConfig(ApiClass):
    connector_type: enums.ConnectorType = dataclasses.field(default=None)


@dataclasses.dataclass
class FileConnectorExportConfig(FeatureGroupExportConfig):
    connector_type: enums.ConnectorType = dataclasses.field(default=enums.ConnectorType.FILE, repr=False)
    location: str = dataclasses.field(default=None)
    export_file_format: str = dataclasses.field(default=None)

    def to_dict(self):
        return {
            'connector_type': self.connector_type,
            'location': self.location,
            'export_file_format': self.export_file_format,
        }


@dataclasses.dataclass
class DatabaseConnectorExportConfig(FeatureGroupExportConfig):
    connector_type: enums.ConnectorType = dataclasses.field(default=enums.ConnectorType.DATABASE, repr=False)
    database_connector_id: str = dataclasses.field(default=None)
    mode: str = dataclasses.field(default=None)
    object_name: str = dataclasses.field(default=None)
    id_column: str = dataclasses.field(default=None)
    additional_id_columns: List[str] = dataclasses.field(default=None)
    data_columns: Dict[str, str] = dataclasses.field(default=None)

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
    config_class_key = 'connectorType'
    config_class_map = {
        enums.ConnectorType.FILE: FileConnectorExportConfig,
        enums.ConnectorType.DATABASE: DatabaseConnectorExportConfig,
    }
