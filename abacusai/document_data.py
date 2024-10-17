from .return_class import AbstractApiClass


class DocumentData(AbstractApiClass):
    """
        Data extracted from a docstore document.

        Args:
            client (ApiClient): An authenticated API Client instance
            docId (str): Unique Docstore string identifier for the document.
            mimeType (str): The mime type of the document.
            pageCount (int): The number of pages for which the data is available. This is generally same as the total number of pages but may be less than the total number of pages in the document if processing is done only for selected pages.
            totalPageCount (int): The total number of pages in the document.
            extractedText (str): The extracted text in the document obtained from OCR.
            embeddedText (str): The embedded text in the document. Only available for digital documents.
            pages (list): List of embedded text for each page in the document. Only available for digital documents.
            tokens (list): List of extracted tokens in the document obtained from OCR.
            metadata (list): List of metadata for each page in the document.
            pageMarkdown (list): The markdown text for the page.
            extractedPageText (list): List of extracted text for each page in the document obtained from OCR. Available when return_extracted_page_text parameter is set to True in the document data retrieval API.
            augmentedPageText (list): List of extracted text for each page in the document obtained from OCR augmented with embedded links in the document.
    """

    def __init__(self, client, docId=None, mimeType=None, pageCount=None, totalPageCount=None, extractedText=None, embeddedText=None, pages=None, tokens=None, metadata=None, pageMarkdown=None, extractedPageText=None, augmentedPageText=None):
        super().__init__(client, None)
        self.doc_id = docId
        self.mime_type = mimeType
        self.page_count = pageCount
        self.total_page_count = totalPageCount
        self.extracted_text = extractedText
        self.embedded_text = embeddedText
        self.pages = pages
        self.tokens = tokens
        self.metadata = metadata
        self.page_markdown = pageMarkdown
        self.extracted_page_text = extractedPageText
        self.augmented_page_text = augmentedPageText
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'doc_id': repr(self.doc_id), f'mime_type': repr(self.mime_type), f'page_count': repr(self.page_count), f'total_page_count': repr(self.total_page_count), f'extracted_text': repr(self.extracted_text), f'embedded_text': repr(
            self.embedded_text), f'pages': repr(self.pages), f'tokens': repr(self.tokens), f'metadata': repr(self.metadata), f'page_markdown': repr(self.page_markdown), f'extracted_page_text': repr(self.extracted_page_text), f'augmented_page_text': repr(self.augmented_page_text)}
        class_name = "DocumentData"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'doc_id': self.doc_id, 'mime_type': self.mime_type, 'page_count': self.page_count, 'total_page_count': self.total_page_count, 'extracted_text': self.extracted_text, 'embedded_text': self.embedded_text,
                'pages': self.pages, 'tokens': self.tokens, 'metadata': self.metadata, 'page_markdown': self.page_markdown, 'extracted_page_text': self.extracted_page_text, 'augmented_page_text': self.augmented_page_text}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
