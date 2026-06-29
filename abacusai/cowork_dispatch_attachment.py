from .return_class import AbstractApiClass


class CoworkDispatchAttachment(AbstractApiClass):
    """
        CoWork desktop file upload result.

        Args:
            client (ApiClient): An authenticated API Client instance
            attachmentId (str): Attachment id for downloadAgentAttachment
            filename (str): Original filename
            mimeType (str): MIME type of the uploaded file
            deploymentId (id): Hashed deployment id used for attachment storage lookup
    """

    def __init__(self, client, attachmentId=None, filename=None, mimeType=None, deploymentId=None):
        super().__init__(client, None)
        self.attachment_id = attachmentId
        self.filename = filename
        self.mime_type = mimeType
        self.deployment_id = deploymentId
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'attachment_id': repr(self.attachment_id), f'filename': repr(
            self.filename), f'mime_type': repr(self.mime_type), f'deployment_id': repr(self.deployment_id)}
        class_name = "CoworkDispatchAttachment"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'attachment_id': self.attachment_id, 'filename': self.filename,
                'mime_type': self.mime_type, 'deployment_id': self.deployment_id}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
