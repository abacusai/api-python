from .return_class import AbstractApiClass


class MediaArtifact(AbstractApiClass):
    """
        Media Artifact

        Args:
            client (ApiClient): An authenticated API Client instance
            mediaArtifactId (id): The ID of the media artifact
            mediaType (str): The type of media (IMAGE, VIDEO, AUDIO)
            info (dict): Metadata (width, height, mime_type, file_size_bytes, generation_model, generation_prompt, generation_params, duration_seconds)
            sourceDeploymentConversationId (id): The source conversation ID
            isFavorited (bool): Whether the artifact is favorited
            isShared (bool): Whether the artifact is shared publicly
            publicUrl (str): The permanent public URL (set after sharing)
            thumbnailUrl (str): Signed thumbnail URL (generated at read time)
            mediaUrl (str): Signed full media URL (generated at read time)
            createdAt (str): The creation timestamp
            updatedAt (str): The last update timestamp
    """

    def __init__(self, client, mediaArtifactId=None, mediaType=None, info=None, sourceDeploymentConversationId=None, isFavorited=None, isShared=None, publicUrl=None, thumbnailUrl=None, mediaUrl=None, createdAt=None, updatedAt=None):
        super().__init__(client, mediaArtifactId)
        self.media_artifact_id = mediaArtifactId
        self.media_type = mediaType
        self.info = info
        self.source_deployment_conversation_id = sourceDeploymentConversationId
        self.is_favorited = isFavorited
        self.is_shared = isShared
        self.public_url = publicUrl
        self.thumbnail_url = thumbnailUrl
        self.media_url = mediaUrl
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'media_artifact_id': repr(self.media_artifact_id), f'media_type': repr(self.media_type), f'info': repr(self.info), f'source_deployment_conversation_id': repr(self.source_deployment_conversation_id), f'is_favorited': repr(
            self.is_favorited), f'is_shared': repr(self.is_shared), f'public_url': repr(self.public_url), f'thumbnail_url': repr(self.thumbnail_url), f'media_url': repr(self.media_url), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at)}
        class_name = "MediaArtifact"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'media_artifact_id': self.media_artifact_id, 'media_type': self.media_type, 'info': self.info, 'source_deployment_conversation_id': self.source_deployment_conversation_id, 'is_favorited': self.is_favorited,
                'is_shared': self.is_shared, 'public_url': self.public_url, 'thumbnail_url': self.thumbnail_url, 'media_url': self.media_url, 'created_at': self.created_at, 'updated_at': self.updated_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
