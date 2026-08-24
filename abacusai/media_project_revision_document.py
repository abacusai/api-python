from .return_class import AbstractApiClass


class MediaProjectRevisionDocument(AbstractApiClass):
    """
        Media Project Revision Document — one checkpoint WITH the document it holds.

        Args:
            client (ApiClient): An authenticated API Client instance
            mediaProjectRevisionId (id): The ID of the revision
            origin (str): Who it was taken for — 'agent', 'user' or 'backup'
            description (str): A short line describing the write it checkpointed, if any
            editorState (dict): The document as it stood at that checkpoint, stored snake_case and returned camelCase
            assetUrls (dict): Map of hashed media_artifact_id -> fresh signed URL, for the media this revision references (set at read time)
            thumbnailUrls (dict): The same for their poster frames, which the media bin draws (set at read time)
            createdAt (str): When it was taken
    """

    def __init__(self, client, mediaProjectRevisionId=None, origin=None, description=None, editorState=None, assetUrls=None, thumbnailUrls=None, createdAt=None):
        super().__init__(client, None)
        self.media_project_revision_id = mediaProjectRevisionId
        self.origin = origin
        self.description = description
        self.editor_state = editorState
        self.asset_urls = assetUrls
        self.thumbnail_urls = thumbnailUrls
        self.created_at = createdAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'media_project_revision_id': repr(self.media_project_revision_id), f'origin': repr(self.origin), f'description': repr(self.description), f'editor_state': repr(
            self.editor_state), f'asset_urls': repr(self.asset_urls), f'thumbnail_urls': repr(self.thumbnail_urls), f'created_at': repr(self.created_at)}
        class_name = "MediaProjectRevisionDocument"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'media_project_revision_id': self.media_project_revision_id, 'origin': self.origin, 'description': self.description,
                'editor_state': self.editor_state, 'asset_urls': self.asset_urls, 'thumbnail_urls': self.thumbnail_urls, 'created_at': self.created_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
