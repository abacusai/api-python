from .return_class import AbstractApiClass


class ModelArtifactsExport(AbstractApiClass):
    """
        A Model Artifacts Export Job

        Args:
            client (ApiClient): An authenticated API Client instance
            modelArtifactsExportId (str): Unique identifier for this export.
            modelVersion (str): Version of the model being exported.
            outputLocation (str): File Connector location the feature group is being written to.
            status (str): Current status of the export.
            createdAt (str): Timestamp at which the export was created (ISO-8601 format).
            exportCompletedAt (str): Timestamp at which the export completed (ISO-8601 format).
            error (str): If `status` is `FAILED`, this field will be populated with an error.
    """

    def __init__(self, client, modelArtifactsExportId=None, modelVersion=None, outputLocation=None, status=None, createdAt=None, exportCompletedAt=None, error=None):
        super().__init__(client, modelArtifactsExportId)
        self.model_artifacts_export_id = modelArtifactsExportId
        self.model_version = modelVersion
        self.output_location = outputLocation
        self.status = status
        self.created_at = createdAt
        self.export_completed_at = exportCompletedAt
        self.error = error
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'model_artifacts_export_id': repr(self.model_artifacts_export_id), f'model_version': repr(self.model_version), f'output_location': repr(
            self.output_location), f'status': repr(self.status), f'created_at': repr(self.created_at), f'export_completed_at': repr(self.export_completed_at), f'error': repr(self.error)}
        class_name = "ModelArtifactsExport"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'model_artifacts_export_id': self.model_artifacts_export_id, 'model_version': self.model_version, 'output_location': self.output_location,
                'status': self.status, 'created_at': self.created_at, 'export_completed_at': self.export_completed_at, 'error': self.error}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            ModelArtifactsExport: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Get the description and status of the model artifacts export.

        Args:
            model_artifacts_export_id (str): A unique string identifier for the export.

        Returns:
            ModelArtifactsExport: Object describing the export and its status.
        """
        return self.client.describe_model_artifacts_export(self.model_artifacts_export_id)

    def wait_for_results(self, timeout=3600):
        """
        A waiting call until model artifacts export is created.

        Args:
            timeout (int): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'EXPORTING'}, timeout=timeout)

    def get_status(self):
        """
        Gets the status of the model artifacts export.

        Returns:
            str: A string describing the status of a model artifacts export (pending, complete, etc.).
        """
        return self.describe().status
