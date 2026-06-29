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
            info (dict): Extensible metadata bag
            assetUrls (dict): Map of hashed media_artifact_id -> fresh signed URL (set by describe at read time)
            createdAt (str): The creation timestamp
            updatedAt (str): The last update timestamp
    """

    def __init__(self, client, mediaProjectId=None, name=None, editorState=None, lastRenderArtifactId=None, info=None, assetUrls=None, createdAt=None, updatedAt=None):
        super().__init__(client, mediaProjectId)
        self.media_project_id = mediaProjectId
        self.name = name
        self.editor_state = editorState
        self.last_render_artifact_id = lastRenderArtifactId
        self.info = info
        self.asset_urls = assetUrls
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'media_project_id': repr(self.media_project_id), f'name': repr(self.name), f'editor_state': repr(self.editor_state), f'last_render_artifact_id': repr(
            self.last_render_artifact_id), f'info': repr(self.info), f'asset_urls': repr(self.asset_urls), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at)}
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
        resp = {'media_project_id': self.media_project_id, 'name': self.name, 'editor_state': self.editor_state, 'last_render_artifact_id':
                self.last_render_artifact_id, 'info': self.info, 'asset_urls': self.asset_urls, 'created_at': self.created_at, 'updated_at': self.updated_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
