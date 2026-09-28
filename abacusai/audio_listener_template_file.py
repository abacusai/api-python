from .return_class import AbstractApiClass


class AudioListenerTemplateFile(AbstractApiClass):
    """
        A reference file of a Listener session template. Only its extracted text is kept; the file itself is not.

        Args:
            client (ApiClient): An authenticated API Client instance
            fileId (str): The file's id.
            clientRef (str): The id the desktop app gave the upload.
            filename (str): The file's name.
            mimeType (str): The file's type.
            sizeBytes (int): The file's size.
            pageCount (int): Pages read, for documents that have pages.
            charCount (int): Characters of text kept.
            truncated (bool): Whether the text was cut to fit the template's 50,000 characters.
            status (str): PROCESSING, READY or FAILED.
            error (str): Why the file failed.
            createdAt (str): When it was added.
    """

    def __init__(self, client, fileId=None, clientRef=None, filename=None, mimeType=None, sizeBytes=None, pageCount=None, charCount=None, truncated=None, status=None, error=None, createdAt=None):
        super().__init__(client, None)
        self.file_id = fileId
        self.client_ref = clientRef
        self.filename = filename
        self.mime_type = mimeType
        self.size_bytes = sizeBytes
        self.page_count = pageCount
        self.char_count = charCount
        self.truncated = truncated
        self.status = status
        self.error = error
        self.created_at = createdAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'file_id': repr(self.file_id), f'client_ref': repr(self.client_ref), f'filename': repr(self.filename), f'mime_type': repr(self.mime_type), f'size_bytes': repr(self.size_bytes), f'page_count': repr(
            self.page_count), f'char_count': repr(self.char_count), f'truncated': repr(self.truncated), f'status': repr(self.status), f'error': repr(self.error), f'created_at': repr(self.created_at)}
        class_name = "AudioListenerTemplateFile"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'file_id': self.file_id, 'client_ref': self.client_ref, 'filename': self.filename, 'mime_type': self.mime_type, 'size_bytes': self.size_bytes,
                'page_count': self.page_count, 'char_count': self.char_count, 'truncated': self.truncated, 'status': self.status, 'error': self.error, 'created_at': self.created_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
