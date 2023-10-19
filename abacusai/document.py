from .document_annotation import DocumentAnnotation
from .return_class import AbstractApiClass


class Document(AbstractApiClass):
    """
        A document in a document store

        Args:
            client (ApiClient): An authenticated API Client instance
            key (str): The key for the document
            fileSize (int): file size for the documnet
            createdAt (str): The timestamp at which the document was created.
            annotations (DocumentAnnotation): the annotations for this document
    """

    def __init__(self, client, key=None, fileSize=None, createdAt=None, annotations={}):
        super().__init__(client, None)
        self.key = key
        self.file_size = fileSize
        self.created_at = createdAt
        self.annotations = client._build_class(DocumentAnnotation, annotations)

    def __repr__(self):
        repr_dict = {f'key': repr(self.key), f'file_size': repr(self.file_size), f'created_at': repr(
            self.created_at), f'annotations': repr(self.annotations)}
        class_name = "Document"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'key': self.key, 'file_size': self.file_size, 'created_at': self.created_at,
                'annotations': self._get_attribute_as_dict(self.annotations)}
        return {key: value for key, value in resp.items() if value is not None}
