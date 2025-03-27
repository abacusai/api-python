import dataclasses

from . import enums
from .abstract import _ApiClassFactory
from .dataset import DatasetConfig, DatasetDocumentProcessingConfig


@dataclasses.dataclass
class ApplicationConnectorDatasetConfig(DatasetConfig):
    """
    An abstract class for dataset configs specific to application connectors.

    Args:
        application_connector_type (enums.ApplicationConnectorType): The type of application connector
        application_connector_id (str): The ID of the application connector
        document_processing_config (DatasetDocumentProcessingConfig): The document processing configuration. Only valid if is_documentset is True for the dataset.
    """
    application_connector_type: enums.ApplicationConnectorType = dataclasses.field(default=None, repr=False, init=False)
    application_connector_id: str = dataclasses.field(default=None)
    document_processing_config: DatasetDocumentProcessingConfig = dataclasses.field(default=None)

    @classmethod
    def _get_builder(cls):
        return _ApplicationConnectorDatasetConfigFactory


@dataclasses.dataclass
class ConfluenceDatasetConfig(ApplicationConnectorDatasetConfig):
    """
    Dataset config for Confluence Application Connector
    Args:
        location (str): The location of the pages to fetch
        space_key (str): The space key of the space from which we fetch pages
        pull_attachments (bool): Whether to pull attachments for each page
        extract_bounding_boxes (bool): Whether to extract bounding boxes from the documents

    """
    location: str = dataclasses.field(default=None)
    space_key: str = dataclasses.field(default=None)
    pull_attachments: bool = dataclasses.field(default=False)
    extract_bounding_boxes: bool = dataclasses.field(default=False)  # TODO: Deprecate in favour of document_processing_config

    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.CONFLUENCE


@dataclasses.dataclass
class BoxDatasetConfig(ApplicationConnectorDatasetConfig):
    """
    Dataset config for Box Application Connector
    Args:
        location (str): The regex location of the files to fetch
        csv_delimiter (str): If the file format is CSV, use a specific csv delimiter
        merge_file_schemas (bool): Signifies if the merge file schema policy is enabled. Not applicable if is_documentset is True
    """
    location: str = dataclasses.field(default=None)
    csv_delimiter: str = dataclasses.field(default=None)
    merge_file_schemas: bool = dataclasses.field(default=False)

    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.BOX


@dataclasses.dataclass
class GoogleAnalyticsDatasetConfig(ApplicationConnectorDatasetConfig):
    """
    Dataset config for Google Analytics Application Connector

    Args:
        location (str): The view id of the report in the connector to fetch
        start_timestamp (int): Unix timestamp of the start of the period that will be queried
        end_timestamp (int): Unix timestamp of the end of the period that will be queried
    """
    location: str = dataclasses.field(default=None)
    start_timestamp: int = dataclasses.field(default=None)
    end_timestamp: int = dataclasses.field(default=None)

    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.GOOGLEANALYTICS


@dataclasses.dataclass
class GoogleDriveDatasetConfig(ApplicationConnectorDatasetConfig):
    """
    Dataset config for Google Drive Application Connector

    Args:
        location (str): The regex location of the files to fetch
        csv_delimiter (str): If the file format is CSV, use a specific csv delimiter
        extract_bounding_boxes (bool): Signifies whether to extract bounding boxes out of the documents. Only valid if is_documentset if True
        merge_file_schemas (bool): Signifies if the merge file schema policy is enabled. Not applicable if is_documentset is True
    """
    location: str = dataclasses.field(default=None)
    csv_delimiter: str = dataclasses.field(default=None)
    extract_bounding_boxes: bool = dataclasses.field(default=False)  # TODO: Deprecate in favour of document_processing_config
    merge_file_schemas: bool = dataclasses.field(default=False)

    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.GOOGLEDRIVE


@dataclasses.dataclass
class JiraDatasetConfig(ApplicationConnectorDatasetConfig):
    """
    Dataset config for Jira Application Connector

    Args:
        jql (str): The JQL query for fetching issues

    """
    jql: str = dataclasses.field(default=None)

    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.JIRA


@dataclasses.dataclass
class OneDriveDatasetConfig(ApplicationConnectorDatasetConfig):
    """
    Dataset config for OneDrive Application Connector

    Args:
        location (str): The regex location of the files to fetch
        csv_delimiter (str): If the file format is CSV, use a specific csv delimiter
        extract_bounding_boxes (bool): Signifies whether to extract bounding boxes out of the documents. Only valid if is_documentset if True
        merge_file_schemas (bool): Signifies if the merge file schema policy is enabled. Not applicable if is_documentset is True
    """
    location: str = dataclasses.field(default=None)
    csv_delimiter: str = dataclasses.field(default=None)
    extract_bounding_boxes: bool = dataclasses.field(default=False)  # TODO: Deprecate in favour of document_processing_config
    merge_file_schemas: bool = dataclasses.field(default=False)

    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.ONEDRIVE


@dataclasses.dataclass
class SharepointDatasetConfig(ApplicationConnectorDatasetConfig):
    """
    Dataset config for Sharepoint Application Connector

    Args:
        location (str): The regex location of the files to fetch
        csv_delimiter (str): If the file format is CSV, use a specific csv delimiter
        extract_bounding_boxes (bool): Signifies whether to extract bounding boxes out of the documents. Only valid if is_documentset if True
        merge_file_schemas (bool): Signifies if the merge file schema policy is enabled. Not applicable if is_documentset is True
    """
    location: str = dataclasses.field(default=None)
    csv_delimiter: str = dataclasses.field(default=None)
    extract_bounding_boxes: bool = dataclasses.field(default=False)  # TODO: Deprecate in favour of document_processing_config
    merge_file_schemas: bool = dataclasses.field(default=False)

    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.SHAREPOINT


@dataclasses.dataclass
class ZendeskDatasetConfig(ApplicationConnectorDatasetConfig):
    """
    Dataset config for Zendesk Application Connector

    Args:
        location (str): The regex location of the files to fetch
    """
    location: str = dataclasses.field(default=None)

    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.ZENDESK


@dataclasses.dataclass
class AbacusUsageMetricsDatasetConfig(ApplicationConnectorDatasetConfig):
    """
    Dataset config for Abacus Usage Metrics Application Connector

    Args:
        include_entire_conversation_history (bool): Whether to show the entire history for this deployment conversation
        include_all_feedback (bool): Whether to include all feedback for this deployment conversation
        resolve_matching_documents (bool): Whether to get matching document references for response instead of prompt.
                                           Needs to recalculate them if highlights are unavailable in summary_info
    """
    include_entire_conversation_history: bool = dataclasses.field(default=False)
    include_all_feedback: bool = dataclasses.field(default=False)
    resolve_matching_documents: bool = dataclasses.field(default=False)

    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.ABACUSUSAGEMETRICS
        self.is_documentset = False


@dataclasses.dataclass
class TeamsScraperDatasetConfig(ApplicationConnectorDatasetConfig):
    """
    Dataset config for Teams Scraper Application Connector

    Args:
        pull_chat_messages (bool): Whether to pull teams chat messages
        pull_channel_posts (bool): Whether to pull posts for each channel
        pull_transcripts (bool): Whether to pull transcripts for calendar meetings
    """
    pull_chat_messages: bool = dataclasses.field(default=False)
    pull_channel_posts: bool = dataclasses.field(default=False)
    pull_transcripts: bool = dataclasses.field(default=False)

    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.TEAMSSCRAPER
        self.is_documentset = True


@dataclasses.dataclass
class FreshserviceDatasetConfig(ApplicationConnectorDatasetConfig):
    """
    Dataset config for Freshservice Application Connector
    """

    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.FRESHSERVICE


@dataclasses.dataclass
class SftpDatasetConfig(ApplicationConnectorDatasetConfig):
    """
    Dataset config for SFTP Application Connector

    Args:
        location (str): The regex location of the files to fetch
        csv_delimiter (str): If the file format is CSV, use a specific csv delimiter
        extract_bounding_boxes (bool): Signifies whether to extract bounding boxes out of the documents. Only valid if is_documentset if True
        merge_file_schemas (bool): Signifies if the merge file schema policy is enabled. Not applicable if is_documentset is True
    """
    location: str = dataclasses.field(default=None)
    csv_delimiter: str = dataclasses.field(default=None)
    extract_bounding_boxes: bool = dataclasses.field(default=False)  # TODO: Deprecate in favour of document_processing_config
    merge_file_schemas: bool = dataclasses.field(default=False)

    def __post_init__(self):
        self.application_connector_type = enums.ApplicationConnectorType.SFTPAPPLICATION


@dataclasses.dataclass
class _ApplicationConnectorDatasetConfigFactory(_ApiClassFactory):
    config_abstract_class = ApplicationConnectorDatasetConfig
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
        enums.ApplicationConnectorType.FRESHSERVICE: FreshserviceDatasetConfig,
        enums.ApplicationConnectorType.TEAMSSCRAPER: TeamsScraperDatasetConfig,
        enums.ApplicationConnectorType.BOX: BoxDatasetConfig,
        enums.ApplicationConnectorType.SFTPAPPLICATION: SftpDatasetConfig,
    }
