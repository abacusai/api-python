from .return_class import AbstractApiClass


class MediaProjectRevision(AbstractApiClass):
    """
        Media Project Revision — one checkpoint in a Studio video project's history.

        Args:
            client (ApiClient): An authenticated API Client instance
            mediaProjectRevisionId (id): The ID of the revision
            origin (str): Who it was taken for — 'agent', 'user' or 'backup'
            description (str): A short line describing the write it checkpointed, if any
            createdAt (str): When it was taken
    """

    def __init__(self, client, mediaProjectRevisionId=None, origin=None, description=None, createdAt=None):
        super().__init__(client, mediaProjectRevisionId)
        self.media_project_revision_id = mediaProjectRevisionId
        self.origin = origin
        self.description = description
        self.created_at = createdAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'media_project_revision_id': repr(self.media_project_revision_id), f'origin': repr(
            self.origin), f'description': repr(self.description), f'created_at': repr(self.created_at)}
        class_name = "MediaProjectRevision"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'media_project_revision_id': self.media_project_revision_id,
                'origin': self.origin, 'description': self.description, 'created_at': self.created_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
