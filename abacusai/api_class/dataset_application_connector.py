import dataclasses

from . import enums
from .abstract import ApiClass, _ApiClassFactory


@dataclasses.dataclass
class DatasetConfig(ApiClass):
    """
    An abstract class for dataset configs specific to application connectors.
    """
    application_connector_type: enums.ApplicationConnectorType = dataclasses.field(default=None, repr=False, init=False)

    @classmethod
    def _get_builder(cls):
        return _DatasetConfigFactory


@dataclasses.dataclass
class ConfluenceDatasetConfig(DatasetConfig):
    """
    Dataset config for Confluence Application Connector
    Args:
        pull_attachments (bool, optional): Whether to pull attachments for each page
        space_key (str, optional): The space key to fetch pages from

    """
    pull_attachments: bool = dataclasses.field(default=False)
    space_key: str = dataclasses.field(default=None)

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
class GoogleDriveDatasetConfig(DatasetConfig):
    """
    Dataset config for Google Drive Application Connector

    Args:
        location (str): The regex location of the files to fetch
        is_documentset (bool): Whether the dataset is a document set
        csv_delimiter (str, optional): If the file format is CSV, use a specific csv delimiter
        extract_bounding_boxes (bool, optional): Signifies whether to extract bounding boxes out of the documents. Only valid if is_documentset if True
        merge_file_schemas (bool, optional): Signifies if the merge file schema policy is enabled. Not applicable if is_documentset is True
    """
    location: str = dataclasses.field(default=None)
    is_documentset: bool = dataclasses.field(default=None)
    csv_delimiter: str = dataclasses.field(default=None)
    extract_bounding_boxes: bool = dataclasses.field(default=False)
    merge_file_schemas: bool = dataclasses.field(default=False)

    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.GOOGLEDRIVE


@dataclasses.dataclass
class JiraDatasetConfig(DatasetConfig):
    """
    Dataset config for Jira Application Connector

    Args:
        jql (str): The JQL query for fetching issues
        custom_fields (list, optional): A list of custom fields to include in the dataset
        include_comments (bool, optional): Fetch comments for each issue
        include_watchers (bool, optional): Fetch watchers for each issue
    """
    jql: str = dataclasses.field(default=None)
    custom_fields: list = dataclasses.field(default=None)
    include_comments: bool = dataclasses.field(default=False)
    include_watchers: bool = dataclasses.field(default=False)

    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.JIRA


@dataclasses.dataclass
class OneDriveDatasetConfig(DatasetConfig):
    """
    Dataset config for OneDrive Application Connector

    Args:
        location (str): The regex location of the files to fetch
        is_documentset (bool): Whether the dataset is a document set
        csv_delimiter (str, optional): If the file format is CSV, use a specific csv delimiter
        extract_bounding_boxes (bool, optional): Signifies whether to extract bounding boxes out of the documents. Only valid if is_documentset if True
        merge_file_schemas (bool, optional): Signifies if the merge file schema policy is enabled. Not applicable if is_documentset is True
    """
    location: str = dataclasses.field(default=None)
    is_documentset: bool = dataclasses.field(default=None)
    csv_delimiter: str = dataclasses.field(default=None)
    extract_bounding_boxes: bool = dataclasses.field(default=False)
    merge_file_schemas: bool = dataclasses.field(default=False)

    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.ONEDRIVE


@dataclasses.dataclass
class SharepointDatasetConfig(DatasetConfig):
    """
    Dataset config for Sharepoint Application Connector

    Args:
        location (str): The regex location of the files to fetch
        is_documentset (bool): Whether the dataset is a document set
        csv_delimiter (str, optional): If the file format is CSV, use a specific csv delimiter
        extract_bounding_boxes (bool, optional): Signifies whether to extract bounding boxes out of the documents. Only valid if is_documentset if True
        merge_file_schemas (bool, optional): Signifies if the merge file schema policy is enabled. Not applicable if is_documentset is True
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
class AbacusUsageMetricsDatasetConfig(DatasetConfig):
    """
    Dataset config for Abacus Usage Metrics Application Connector

    Args:
        include_entire_conversation_history (bool): Whether to show the entire history for this deployment conversation
    """
    include_entire_conversation_history: bool = dataclasses.field(default=False)

    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.ABACUSUSAGEMETRICS


@dataclasses.dataclass
class _DatasetConfigFactory(_ApiClassFactory):
    config_abstract_class = DatasetConfig
    config_class_key = 'application_connector_type'
    config_class_map = {
        enums.ApplicationConnectorType.CONFLUENCE: ConfluenceDatasetConfig,
        enums.ApplicationConnectorType.GOOGLEANALYTICS: GoogleAnalyticsDatasetConfig,
        enums.ApplicationConnectorType.GOOGLEDRIVE: GoogleDriveDatasetConfig,
        enums.ApplicationConnectorType.JIRA: JiraDatasetConfig,
        enums.ApplicationConnectorType.ONEDRIVE: OneDriveDatasetConfig,
        enums.ApplicationConnectorType.SHAREPOINT: SharepointDatasetConfig,
        enums.ApplicationConnectorType.ZENDESK: ZendeskDatasetConfig,
        enums.ApplicationConnectorType.ABACUSUSAGEMETRICS: AbacusUsageMetricsDatasetConfig,
    }
