from .return_class import AbstractApiClass


class DocumentStore(AbstractApiClass):
    """
        A document store.

        Args:
            client (ApiClient): An authenticated API Client instance
            documentStoreId (str): A unique string identifier for the document store.
            createdAt (str): The timestamp in ISO-8601 format when the document store was created.
            name (str): The name of the document store.
            documentType (str): The type of documents stored in the document store, as an enumerated string.
            documentCount (int): The number of documents in the document store.
            approximateSize (int): An approximate count of bytes for all documents stored in the document store.
    """

    def __init__(self, client, documentStoreId=None, createdAt=None, name=None, documentType=None, documentCount=None, approximateSize=None):
        super().__init__(client, documentStoreId)
        self.document_store_id = documentStoreId
        self.created_at = createdAt
        self.name = name
        self.document_type = documentType
        self.document_count = documentCount
        self.approximate_size = approximateSize

    def __repr__(self):
        repr_dict = {f'document_store_id': repr(self.document_store_id), f'created_at': repr(self.created_at), f'name': repr(self.name), f'document_type': repr(
            self.document_type), f'document_count': repr(self.document_count), f'approximate_size': repr(self.approximate_size)}
        class_name = "DocumentStore"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'document_store_id': self.document_store_id, 'created_at': self.created_at, 'name': self.name,
                'document_type': self.document_type, 'document_count': self.document_count, 'approximate_size': self.approximate_size}
        return {key: value for key, value in resp.items() if value is not None}
