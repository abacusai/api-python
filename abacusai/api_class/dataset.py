import dataclasses

from .abstract import ApiClass
from .enums import OcrMode


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
        extract_bounding_boxes (bool): Whether to perform OCR and extract bounding boxes. If False, no OCR will be done but only the embedded text from digital documents will be extracted. Defaults to False.
        ocr_mode (OcrMode): OCR mode. There are different OCR modes available for different kinds of documents and use cases. This option only takes effect when extract_bounding_boxes is True.
        use_full_ocr (bool): Whether to perform full OCR. If True, OCR will be performed on the full page. If False, OCR will be performed on the non-text regions only. By default, it will be decided automatically based on the OCR mode and the document type. This option only takes effect when extract_bounding_boxes is True.
        remove_header_footer (bool): Whether to remove headers and footers. Defaults to False. This option only takes effect when extract_bounding_boxes is True.
        remove_watermarks (bool): Whether to remove watermarks. By default, it will be decided automatically based on the OCR mode and the document type. This option only takes effect when extract_bounding_boxes is True.
        convert_to_markdown (bool): Whether to convert extracted text to markdown. Defaults to False. This option only takes effect when extract_bounding_boxes is True.
    """
    # NOTE: The defaults should match with clouddb.document_processing_results table defaults
    extract_bounding_boxes: bool = False
    ocr_mode: OcrMode = OcrMode.DEFAULT
    use_full_ocr: bool = None
    remove_header_footer: bool = False
    remove_watermarks: bool = True
    convert_to_markdown: bool = False


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
