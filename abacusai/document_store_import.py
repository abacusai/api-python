from .return_class import AbstractApiClass


class DocumentStoreImport(AbstractApiClass):
    """
        A document store import

        Args:
            client (ApiClient): An authenticated API Client instance
            documentStoreImportId (str): The unique identifier of the import
            documentStoreId (str): The unique identifier of the notebook.
            importingStartedAt (str): The timestamp at which the notebook was created.
            status (str): The current status of the dataset version
            error (str): If status is FAILED, this field will be populated with an error.
            documentsImported (int): The number of documents in the document store
            documentsSkipped (int): The number of documents in the document store
            uploadId (str): The unique identifier to be used for documents upload
            replaceExistingFiles (bool): If false, documents that match the same key path in the document store will be skipped
    """

    def __init__(self, client, documentStoreImportId=None, documentStoreId=None, importingStartedAt=None, status=None, error=None, documentsImported=None, documentsSkipped=None, uploadId=None, replaceExistingFiles=None):
        super().__init__(client, documentStoreImportId)
        self.document_store_import_id = documentStoreImportId
        self.document_store_id = documentStoreId
        self.importing_started_at = importingStartedAt
        self.status = status
        self.error = error
        self.documents_imported = documentsImported
        self.documents_skipped = documentsSkipped
        self.upload_id = uploadId
        self.replace_existing_files = replaceExistingFiles

    def __repr__(self):
        return f"DocumentStoreImport(document_store_import_id={repr(self.document_store_import_id)},\n  document_store_id={repr(self.document_store_id)},\n  importing_started_at={repr(self.importing_started_at)},\n  status={repr(self.status)},\n  error={repr(self.error)},\n  documents_imported={repr(self.documents_imported)},\n  documents_skipped={repr(self.documents_skipped)},\n  upload_id={repr(self.upload_id)},\n  replace_existing_files={repr(self.replace_existing_files)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'document_store_import_id': self.document_store_import_id, 'document_store_id': self.document_store_id, 'importing_started_at': self.importing_started_at, 'status': self.status, 'error': self.error, 'documents_imported': self.documents_imported, 'documents_skipped': self.documents_skipped, 'upload_id': self.upload_id, 'replace_existing_files': self.replace_existing_files}
