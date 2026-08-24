from .return_class import AbstractApiClass


class MediaProject(AbstractApiClass):
    """
        Media Project — a saved Studio video-editor project.

        Args:
            client (ApiClient): An authenticated API Client instance
            mediaProjectId (id): The ID of the media project
            name (str): The project name
            editorState (dict): The editor document (settings + media library + timeline), stored snake_case and returned camelCase
            lastRenderArtifactId (id): The media artifact of the latest export, if any
            sourceDeploymentConversationId (id): The conversation this project was handed off from, if any
            info (dict): Extensible metadata bag
            assetUrls (dict): Map of hashed media_artifact_id -> fresh signed URL (set by describe at read time)
            thumbnailUrls (dict): Map of hashed media_artifact_id -> fresh signed thumbnail URL (set by describe at read time)
            thumbnailUrl (str): Fresh signed cover image for the project (latest export or first video clip), set at read time
            lastRenderUrl (str): Fresh signed URL of the latest export itself, set at read time. Distinct from thumbnail_url, which may be an explicit cover or a source clip rather than the export
            lastRenderThumbnailUrl (str): Fresh signed poster frame for that export, set at read time
            createdAt (str): The creation timestamp
            updatedAt (str): The last update timestamp
    """

    def __init__(self, client, mediaProjectId=None, name=None, editorState=None, lastRenderArtifactId=None, sourceDeploymentConversationId=None, info=None, assetUrls=None, thumbnailUrls=None, thumbnailUrl=None, lastRenderUrl=None, lastRenderThumbnailUrl=None, createdAt=None, updatedAt=None):
        super().__init__(client, mediaProjectId)
        self.media_project_id = mediaProjectId
        self.name = name
        self.editor_state = editorState
        self.last_render_artifact_id = lastRenderArtifactId
        self.source_deployment_conversation_id = sourceDeploymentConversationId
        self.info = info
        self.asset_urls = assetUrls
        self.thumbnail_urls = thumbnailUrls
        self.thumbnail_url = thumbnailUrl
        self.last_render_url = lastRenderUrl
        self.last_render_thumbnail_url = lastRenderThumbnailUrl
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'media_project_id': repr(self.media_project_id), f'name': repr(self.name), f'editor_state': repr(self.editor_state), f'last_render_artifact_id': repr(self.last_render_artifact_id), f'source_deployment_conversation_id': repr(self.source_deployment_conversation_id), f'info': repr(self.info), f'asset_urls': repr(
            self.asset_urls), f'thumbnail_urls': repr(self.thumbnail_urls), f'thumbnail_url': repr(self.thumbnail_url), f'last_render_url': repr(self.last_render_url), f'last_render_thumbnail_url': repr(self.last_render_thumbnail_url), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at)}
        class_name = "MediaProject"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'media_project_id': self.media_project_id, 'name': self.name, 'editor_state': self.editor_state, 'last_render_artifact_id': self.last_render_artifact_id, 'source_deployment_conversation_id': self.source_deployment_conversation_id, 'info': self.info,
                'asset_urls': self.asset_urls, 'thumbnail_urls': self.thumbnail_urls, 'thumbnail_url': self.thumbnail_url, 'last_render_url': self.last_render_url, 'last_render_thumbnail_url': self.last_render_thumbnail_url, 'created_at': self.created_at, 'updated_at': self.updated_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
