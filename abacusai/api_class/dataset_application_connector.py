import dataclasses

from . import enums
from .abstract import ApiClass, _ApiClassFactory


@dataclasses.dataclass
class DatasetConfig(ApiClass):
    application_connector_type: enums.ApplicationConnectorType = dataclasses.field(default=None, repr=False)

    @classmethod
    def _get_builder(cls):
        return _DatasetConfigFactory


@dataclasses.dataclass
class ConfluenceDatasetConfig(DatasetConfig):
    """
    Dataset config for Confluence Application Connector
    """
    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.CONFLUENCE


@dataclasses.dataclass
class GoogleAnalyticsDatasetConfig(DatasetConfig):
    """
    Dataset config for Google Analytics Application Connector
    Args:
        location (str): The view id of the report in the connector to fetch
        start_timestamp (int, optional): Unix timestamp of the start of the period that will be queried
        end_timestamp (int, optional): Unix timestamp of the end of the period that will be queried
    """
    location: str = dataclasses.field(default=None)
    start_timestamp: int = dataclasses.field(default=None)
    end_timestamp: int = dataclasses.field(default=None)

    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.GOOGLEANALYTICS


@dataclasses.dataclass
class SharepointDatasetConfig(DatasetConfig):
    """
    Dataset config for Sharepoint Application Connector
    Args:
        location (str): The regex location of the files to fetch
        is_documentset (bool): Whether the dataset is a document set
        csv_delimiter (str, optional): If the file format is CSV, use a specific csv delimiter
        extract_bounding_boxes (bool, optional): Signifies whether to extract bounding boxes out of the documents. Only valid if is_documentset if True
        merge_file_schemas (bool, optional): Signifies if the merge file schema policy is enabled. If is_documentset is True, this is also set to True by default
    """
    location: str = dataclasses.field(default=None)
    is_documentset: bool = dataclasses.field(default=None)
    csv_delimiter: str = dataclasses.field(default=None)
    extract_bounding_boxes: bool = dataclasses.field(default=False)
    merge_file_schemas: bool = dataclasses.field(default=False)

    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.SHAREPOINT


@dataclasses.dataclass
class ZendeskDatasetConfig(DatasetConfig):
    """
    Dataset config for Zendesk Application Connector
    """
    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.ZENDESK


@dataclasses.dataclass
class _DatasetConfigFactory(_ApiClassFactory):
    config_abstract_class = DatasetConfig
    config_class_key = 'applicationConnectorType'
    config_class_map = {
        enums.ApplicationConnectorType.CONFLUENCE: ConfluenceDatasetConfig,
        enums.ApplicationConnectorType.GOOGLEANALYTICS: GoogleAnalyticsDatasetConfig,
        enums.ApplicationConnectorType.SHAREPOINT: SharepointDatasetConfig,
        enums.ApplicationConnectorType.ZENDESK: ZendeskDatasetConfig,
    }
