from .return_class import AbstractApiClass


class AgentDataDocumentInfo(AbstractApiClass):
    """
        Information for documents uploaded to agents.

        Args:
            client (ApiClient): An authenticated API Client instance
            docId (str): The docstore Document ID of the document.
            filename (str): The file name of the uploaded document.
            mimeType (str): The mime type of the uploaded document.
            size (int): The total size of the uploaded document.
            pageCount (int): The total number of pages in the uploaded document.
    """

    def __init__(self, client, docId=None, filename=None, mimeType=None, size=None, pageCount=None):
        super().__init__(client, None)
        self.doc_id = docId
        self.filename = filename
        self.mime_type = mimeType
        self.size = size
        self.page_count = pageCount
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'doc_id': repr(self.doc_id), f'filename': repr(self.filename), f'mime_type': repr(
            self.mime_type), f'size': repr(self.size), f'page_count': repr(self.page_count)}
        class_name = "AgentDataDocumentInfo"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'doc_id': self.doc_id, 'filename': self.filename,
                'mime_type': self.mime_type, 'size': self.size, 'page_count': self.page_count}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
