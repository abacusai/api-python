from .return_class import AbstractApiClass


class DocumentData(AbstractApiClass):
    """
        Data extracted from a docstore document.

        Args:
            client (ApiClient): An authenticated API Client instance
            docId (str): Unique Docstore string identifier for the document.
            mimeType (str): The mime type of the document.
            pageCount (int): The total number of pages in document.
            extractedText (str): The extracted text in the document obtained from OCR.
            embeddedText (str): The embedded text in the document. Only available for digital documents.
            pages (list): List of embedded text for each page in the document. Only available for digital documents.
            tokens (list): List of extracted tokens in the document obtained from OCR.
            metadata (list): List of metadata for each page in the document.
            pageMarkdown (list): The markdown text for the page.
    """

    def __init__(self, client, docId=None, mimeType=None, pageCount=None, extractedText=None, embeddedText=None, pages=None, tokens=None, metadata=None, pageMarkdown=None):
        super().__init__(client, None)
        self.doc_id = docId
        self.mime_type = mimeType
        self.page_count = pageCount
        self.extracted_text = extractedText
        self.embedded_text = embeddedText
        self.pages = pages
        self.tokens = tokens
        self.metadata = metadata
        self.page_markdown = pageMarkdown
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'doc_id': repr(self.doc_id), f'mime_type': repr(self.mime_type), f'page_count': repr(self.page_count), f'extracted_text': repr(self.extracted_text), f'embedded_text': repr(
            self.embedded_text), f'pages': repr(self.pages), f'tokens': repr(self.tokens), f'metadata': repr(self.metadata), f'page_markdown': repr(self.page_markdown)}
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
        resp = {'doc_id': self.doc_id, 'mime_type': self.mime_type, 'page_count': self.page_count, 'extracted_text': self.extracted_text,
                'embedded_text': self.embedded_text, 'pages': self.pages, 'tokens': self.tokens, 'metadata': self.metadata, 'page_markdown': self.page_markdown}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
