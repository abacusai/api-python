from .return_class import AbstractApiClass


class Avatar(AbstractApiClass):
    """
        Avatar — a persistent, reusable identity (reference image + optional cloned voice) for Studio generations.

        Args:
            client (ApiClient): An authenticated API Client instance
            avatarId (id): The ID of the avatar
            name (str): The avatar's display name
            description (str): Persona/descriptor used by the ad script writer and shown in the detail dialog
            isStock (bool): Whether this is a seeded starter avatar (read-only, visible to every org)
            isFavorited (bool): Whether the avatar is favorited
            voiceStatus (str): Voice lifecycle state (none, processing, ready, failed), derived at read time
            voicePreviewUrl (str): Signed URL of the stored voice sample when one exists, generated at read time
            imageUrls (list): Full-size signed URLs of the reference images (first = the front reference, the one the cards show), generated at read time
            thumbnailUrls (list): Signed URLs of the card-sized copies, same order as image_urls (falls back to the full image where there is none)
            createdAt (str): The creation timestamp
            updatedAt (str): The last update timestamp
    """

    def __init__(self, client, avatarId=None, name=None, description=None, isStock=None, isFavorited=None, voiceStatus=None, voicePreviewUrl=None, imageUrls=None, thumbnailUrls=None, createdAt=None, updatedAt=None):
        super().__init__(client, avatarId)
        self.avatar_id = avatarId
        self.name = name
        self.description = description
        self.is_stock = isStock
        self.is_favorited = isFavorited
        self.voice_status = voiceStatus
        self.voice_preview_url = voicePreviewUrl
        self.image_urls = imageUrls
        self.thumbnail_urls = thumbnailUrls
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'avatar_id': repr(self.avatar_id), f'name': repr(self.name), f'description': repr(self.description), f'is_stock': repr(self.is_stock), f'is_favorited': repr(self.is_favorited), f'voice_status': repr(
            self.voice_status), f'voice_preview_url': repr(self.voice_preview_url), f'image_urls': repr(self.image_urls), f'thumbnail_urls': repr(self.thumbnail_urls), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at)}
        class_name = "Avatar"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'avatar_id': self.avatar_id, 'name': self.name, 'description': self.description, 'is_stock': self.is_stock, 'is_favorited': self.is_favorited, 'voice_status': self.voice_status,
                'voice_preview_url': self.voice_preview_url, 'image_urls': self.image_urls, 'thumbnail_urls': self.thumbnail_urls, 'created_at': self.created_at, 'updated_at': self.updated_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
