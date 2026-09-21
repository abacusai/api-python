from .return_class import AbstractApiClass


class AdTemplate(AbstractApiClass):
    """
        Ad Template — a Studio ad format card (client-safe registry fields; recipes stay server-side).

        Args:
            client (ApiClient): An authenticated API Client instance
            slug (str): Stable template identifier
            name (str): Display name
            category (str): Catalog category — one of ugc, tiktok, commercial
            supportsAvatar (bool): Whether an avatar can be attached (some formats are product-only)
            needsProduct (bool): False for avatar-only social formats (no product; avatar required). Absent means True
            defaultDuration (int): Suggested duration in seconds
            defaultAspectRatio (str): Suggested aspect ratio
            cardOneLiner (str): One-line pitch on the format card
            cardDescription (str): Two-line description of the video the format makes, for its preview dialog
            description (str): Gallery caption / display copy
            exampleMedia (list): CDN URLs of preview videos for the card
            exampleMediaPreview (list): Compressed 480p encodes of example_media, for the card walls
    """

    def __init__(self, client, slug=None, name=None, category=None, supportsAvatar=None, needsProduct=None, defaultDuration=None, defaultAspectRatio=None, cardOneLiner=None, cardDescription=None, description=None, exampleMedia=None, exampleMediaPreview=None):
        super().__init__(client, None)
        self.slug = slug
        self.name = name
        self.category = category
        self.supports_avatar = supportsAvatar
        self.needs_product = needsProduct
        self.default_duration = defaultDuration
        self.default_aspect_ratio = defaultAspectRatio
        self.card_one_liner = cardOneLiner
        self.card_description = cardDescription
        self.description = description
        self.example_media = exampleMedia
        self.example_media_preview = exampleMediaPreview
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'slug': repr(self.slug), f'name': repr(self.name), f'category': repr(self.category), f'supports_avatar': repr(self.supports_avatar), f'needs_product': repr(self.needs_product), f'default_duration': repr(self.default_duration), f'default_aspect_ratio': repr(
            self.default_aspect_ratio), f'card_one_liner': repr(self.card_one_liner), f'card_description': repr(self.card_description), f'description': repr(self.description), f'example_media': repr(self.example_media), f'example_media_preview': repr(self.example_media_preview)}
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
        resp = {'slug': self.slug, 'name': self.name, 'category': self.category, 'supports_avatar': self.supports_avatar, 'needs_product': self.needs_product, 'default_duration': self.default_duration, 'default_aspect_ratio': self.default_aspect_ratio,
                'card_one_liner': self.card_one_liner, 'card_description': self.card_description, 'description': self.description, 'example_media': self.example_media, 'example_media_preview': self.example_media_preview}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
