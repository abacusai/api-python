from .document_annotation import DocumentAnnotation
from .return_class import AbstractApiClass


class Document(AbstractApiClass):
    """
        A document in a document store

        Args:
            client (ApiClient): An authenticated API Client instance
            key (str): 
            fileSize (int): 
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
        return f"Document(key={repr(self.key)},\n  file_size={repr(self.file_size)},\n  created_at={repr(self.created_at)},\n  annotations={repr(self.annotations)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'key': self.key, 'file_size': self.file_size, 'created_at': self.created_at, 'annotations': self._get_attribute_as_dict(self.annotations)}
