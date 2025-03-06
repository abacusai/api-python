import dataclasses

from .abstract import ApiClass
from .enums import DocumentType, OcrMode


@dataclasses.dataclass
class DatasetConfig(ApiClass):
    """
    An abstract class for dataset configs

    Args:
        is_documentset (bool): Whether the dataset is a document set
    """
    is_documentset: bool = dataclasses.field(default=None)


@dataclasses.dataclass
class ParsingConfig(ApiClass):
    """
    Custom config for dataset parsing.

    Args:
        escape (str): Escape character for CSV files. Defaults to '"'.
        csv_delimiter (str): Delimiter for CSV files. Defaults to None.
        file_path_with_schema (str): Path to the file with schema. Defaults to None.
    """
    escape: str = dataclasses.field(default='"')
    csv_delimiter: str = dataclasses.field(default=None)
    file_path_with_schema: str = dataclasses.field(default=None)


@dataclasses.dataclass
class DocumentProcessingConfig(ApiClass):
    """
    Document processing configuration.

    Args:
        document_type (DocumentType): Type of document. Can be one of Text, Tables and Forms, Embedded Images, etc. If not specified, type will be decided automatically.
        highlight_relevant_text (bool): Whether to extract bounding boxes and highlight relevant text in search results. Defaults to False.
        extract_bounding_boxes (bool): Whether to perform OCR and extract bounding boxes. If False, no OCR will be done but only the embedded text from digital documents will be extracted. Defaults to False.
        ocr_mode (OcrMode): OCR mode. There are different OCR modes available for different kinds of documents and use cases. This option only takes effect when extract_bounding_boxes is True.
        use_full_ocr (bool): Whether to perform full OCR. If True, OCR will be performed on the full page. If False, OCR will be performed on the non-text regions only. By default, it will be decided automatically based on the OCR mode and the document type. This option only takes effect when extract_bounding_boxes is True.
        remove_header_footer (bool): Whether to remove headers and footers. Defaults to False. This option only takes effect when extract_bounding_boxes is True.
        remove_watermarks (bool): Whether to remove watermarks. By default, it will be decided automatically based on the OCR mode and the document type. This option only takes effect when extract_bounding_boxes is True.
        convert_to_markdown (bool): Whether to convert extracted text to markdown. Defaults to False. This option only takes effect when extract_bounding_boxes is True.
        mask_pii (bool): Whether to mask personally identifiable information (PII) in the document text/tokens. Defaults to False.
        extract_images (bool): Whether to extract images from the document e.g. diagrams in a PDF page. Defaults to False.
    """
    # NOTE: The defaults should match with clouddb.document_processing_results table defaults
    document_type: DocumentType = None
    highlight_relevant_text: bool = None
    extract_bounding_boxes: bool = False
    ocr_mode: OcrMode = OcrMode.DEFAULT
    use_full_ocr: bool = None
    remove_header_footer: bool = False
    remove_watermarks: bool = True
    convert_to_markdown: bool = False
    mask_pii: bool = False
    extract_images: bool = False

    def __post_init__(self):
        self.ocr_mode = self._detect_ocr_mode()
        if self.document_type is not None:
            if DocumentType.is_ocr_forced(self.document_type):
                self.highlight_relevant_text = True
            else:
                self.highlight_relevant_text = False
        if self.highlight_relevant_text is not None:
            self.extract_bounding_boxes = self.highlight_relevant_text  # Highlight_relevant text acts as a wrapper over extract_bounding_boxes

    def _detect_ocr_mode(self):
        if self.document_type is not None:
            if self.document_type == DocumentType.TEXT or self.document_type == DocumentType.SIMPLE_TEXT:
                return OcrMode.DEFAULT
            elif self.document_type == DocumentType.TABLES_AND_FORMS:
                return OcrMode.LAYOUT
            elif self.document_type == DocumentType.COMPREHENSIVE_MARKDOWN:
                return OcrMode.COMPREHENSIVE_TABLE_MD
            elif self.document_type == DocumentType.EMBEDDED_IMAGES:
                return OcrMode.SCANNED
            elif self.document_type == DocumentType.SCANNED_TEXT:
                return OcrMode.SCANNED
        if self.ocr_mode is not None:
            return self.ocr_mode
        return OcrMode.AUTO

    @classmethod
    def _get_filtered_dict(cls, config: dict):
        """Filters out default values from the config"""
        from reainternal.utils import snake_case
        return {
            k: v for k, v in config.items()
            if v is not None and v != getattr(cls, snake_case(k), None)
        }


@dataclasses.dataclass
class DatasetDocumentProcessingConfig(DocumentProcessingConfig):
    """
    Document processing configuration for dataset imports.

    Args:
        extract_bounding_boxes (bool): Whether to perform OCR and extract bounding boxes. If False, no OCR will be done but only the embedded text from digital documents will be extracted. Defaults to False.
        ocr_mode (OcrMode): OCR mode. There are different OCR modes available for different kinds of documents and use cases. This option only takes effect when extract_bounding_boxes is True.
        use_full_ocr (bool): Whether to perform full OCR. If True, OCR will be performed on the full page. If False, OCR will be performed on the non-text regions only. By default, it will be decided automatically based on the OCR mode and the document type. This option only takes effect when extract_bounding_boxes is True.
        remove_header_footer (bool): Whether to remove headers and footers. Defaults to False. This option only takes effect when extract_bounding_boxes is True.
        remove_watermarks (bool): Whether to remove watermarks. By default, it will be decided automatically based on the OCR mode and the document type. This option only takes effect when extract_bounding_boxes is True.
        convert_to_markdown (bool): Whether to convert extracted text to markdown. Defaults to False. This option only takes effect when extract_bounding_boxes is True.
        page_text_column (str): Name of the output column which contains the extracted text for each page. If not provided, no column will be created.
    """
    page_text_column: str = None


@dataclasses.dataclass
class IncrementalDatabaseConnectorConfig(ApiClass):
    """
    Config information for incremental datasets from database connectors

    Args:
        timestamp_column (str): If dataset is incremental, this is the column name of the required column in the dataset. This column must contain timestamps in descending order which are used to determine the increments of the incremental dataset.
    """
    timestamp_column: str = dataclasses.field(default=None)


@dataclasses.dataclass
class AttachmentParsingConfig(ApiClass):
    """
    Config information for parsing attachments

    Args:
        feature_group_name (str): feature group name
        column_name (str): column name
        urls (str): list of urls
    """
    feature_group_name: str = dataclasses.field(default=None)
    column_name: str = dataclasses.field(default=None)
    urls: str = dataclasses.field(default=None)
