from .return_class import AbstractApiClass


class ChatSegmentArtifact(AbstractApiClass):
    """
        Shared Code/Artifact Segment

        Args:
            client (ApiClient): An authenticated API Client instance
            artifactId (id): The ID of the chat segment artifact
            artifactMetadata (dict): Metadata (identifier, title, componentName, language, readonly, etc.)
            artifactType (str): The artifact type (SQL, PYTHON, REACT, JSX, MARKDOWN, HTML, SVG, XML, CSS, JSON, JAVASCRIPT, TEXT)
            storageType (str): Where content is stored ('inline' or 's3')
            content (str): Inline artifact content (present when storage_type='inline')
            renderingContent (str): Rendering code for ReactComponents, if applicable
            publicUrl (str): CDN URL for S3-backed content (present when storage_type='s3')
            createdAt (str): The creation timestamp
            updatedAt (str): The last update timestamp
    """

    def __init__(self, client, artifactId=None, artifactMetadata=None, artifactType=None, storageType=None, content=None, renderingContent=None, publicUrl=None, createdAt=None, updatedAt=None):
        super().__init__(client, None)
        self.artifact_id = artifactId
        self.artifact_metadata = artifactMetadata
        self.artifact_type = artifactType
        self.storage_type = storageType
        self.content = content
        self.rendering_content = renderingContent
        self.public_url = publicUrl
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'artifact_id': repr(self.artifact_id), f'artifact_metadata': repr(self.artifact_metadata), f'artifact_type': repr(self.artifact_type), f'storage_type': repr(self.storage_type), f'content': repr(
            self.content), f'rendering_content': repr(self.rendering_content), f'public_url': repr(self.public_url), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at)}
        class_name = "ChatSegmentArtifact"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'artifact_id': self.artifact_id, 'artifact_metadata': self.artifact_metadata, 'artifact_type': self.artifact_type, 'storage_type': self.storage_type,
                'content': self.content, 'rendering_content': self.rendering_content, 'public_url': self.public_url, 'created_at': self.created_at, 'updated_at': self.updated_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
