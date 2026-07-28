from .return_class import AbstractApiClass


class AdTemplate(AbstractApiClass):
    """
        Ad Template — a Studio ad format card (client-safe registry fields; recipes stay server-side).

        Args:
            client (ApiClient): An authenticated API Client instance
            slug (str): Stable template identifier
            name (str): Display name
            category (str): Explore subtab (ugc, tiktok, commercial)
            speaks (bool): Whether the avatar speaks (voice/lip-sync formats)
            supportsAvatar (bool): Whether an avatar can be attached (some formats are product-only)
            defaultDuration (int): Suggested duration in seconds
            defaultAspectRatio (str): Suggested aspect ratio
            cardOneLiner (str): One-line pitch on the format card
            description (str): Gallery caption / display copy
            exampleMedia (list): CDN URLs of preview videos for the card
    """

    def __init__(self, client, slug=None, name=None, category=None, speaks=None, supportsAvatar=None, defaultDuration=None, defaultAspectRatio=None, cardOneLiner=None, description=None, exampleMedia=None):
        super().__init__(client, None)
        self.slug = slug
        self.name = name
        self.category = category
        self.speaks = speaks
        self.supports_avatar = supportsAvatar
        self.default_duration = defaultDuration
        self.default_aspect_ratio = defaultAspectRatio
        self.card_one_liner = cardOneLiner
        self.description = description
        self.example_media = exampleMedia
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'slug': repr(self.slug), f'name': repr(self.name), f'category': repr(self.category), f'speaks': repr(self.speaks), f'supports_avatar': repr(self.supports_avatar), f'default_duration': repr(
            self.default_duration), f'default_aspect_ratio': repr(self.default_aspect_ratio), f'card_one_liner': repr(self.card_one_liner), f'description': repr(self.description), f'example_media': repr(self.example_media)}
        class_name = "AdTemplate"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'slug': self.slug, 'name': self.name, 'category': self.category, 'speaks': self.speaks, 'supports_avatar': self.supports_avatar, 'default_duration': self.default_duration,
                'default_aspect_ratio': self.default_aspect_ratio, 'card_one_liner': self.card_one_liner, 'description': self.description, 'example_media': self.example_media}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
