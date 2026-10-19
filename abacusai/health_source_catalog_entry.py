from .return_class import AbstractApiClass


class HealthSourceCatalogEntry(AbstractApiClass):
    """
        One provider the Health Data Sources surface can show — what it is called, how it looks, and how (or

        Args:
            client (ApiClient): An authenticated API Client instance
            provider (str): Provider key (oura, fitbit, garmin, ...), matching HealthDataSource.provider.
            name (str): Brand name, shown as-is (never translated). What a reading, workout or insight from this provider is attributed to.
            connectName (str): What the connect tile and the connected-source row call it, as-is. Equal to `name` unless the brand distinguishes its app from its devices: Garmin's guidelines want the connection presented as "Garmin Connect" and the data attributed to "Garmin".
            category (str): Stable display key the client maps to a translated label (Wearable, Fitness, Smart scale, Nutrition, Medical, CGM).
            kind (str): wearable | nutrition | cgm — the ingestion-side data shape, not a user-facing label.
            color (str): Hex brand colour for the circular source badge.
            glyph (str): One or two characters drawn on that badge when no icon renders.
            iconUrl (str): For icon_style `mark`, a WHITE monochrome mark on transparency, to be composited over `color`. One asset works in light and dark, on web and native. For icon_style `tile`, the provider's own app tile, to be drawn exactly as served. Absent for a provider whose icon isn't published yet — draw `glyph` instead.
            iconStyle (str): mark (the default: the white mark on a `color` disc, the shape every badge and disc stack draws) | tile (the provider's app tile, drawn unaltered as a rounded square and never tinted, cropped to a circle or placed in a disc stack — Garmin's brand guidelines forbid its mark in avatars and badges, so a client names such a provider in text where it would have drawn a disc).
            attributionRequired (bool): The provider's brand terms require its name beside every view of its data, however many devices the user has (Garmin: "Garmin [device model]" adjacent to each data title). A client otherwise names the device only when there is more than one to tell apart; with any connected provider carrying this it names them always. Always present.
            connectMode (str): web_oauth (Terra's hosted OAuth from a browser) | mobile_app (on-device SDK only — the phone health stores). Derived from the same provider tuple the connect endpoint enforces, so a client that offers what this says can never be refused for offering the wrong thing.
            primaryRank (int): Tie-break order when several connected devices report the SAME metric and are otherwise equally good witnesses — lower wins, unranked providers sort last. Served so the insight cards and a client's metrics view attribute a reading to the same device.
            comingSoon (bool): Held back from launch — list it, but offer no way to connect it. Decided by the server alone (source_catalog.COMING_SOON_PROVIDERS), so web and app agree on what has shipped.
            deprecating (bool): Supported, but the provider is winding the integration down (Google Fit's REST API in favour of Health Connect) — best-effort.
            directUpload (bool): For a mobile_app provider, whether THIS org's app connects it on our own pipeline — the app reads the phone's health store and uploads to _uploadHealthKitSamples — rather than through Terra's on-device SDK. Per org (health_healthkit_direct); always present, false for every other provider and for every org not on the rollout.
            mobileOnlyDetail (str): Why a browser can't connect this one, for a web user. Present only when connect_mode is mobile_app.
    """

    def __init__(self, client, provider=None, name=None, connectName=None, category=None, kind=None, color=None, glyph=None, iconUrl=None, iconStyle=None, attributionRequired=None, connectMode=None, primaryRank=None, comingSoon=None, deprecating=None, directUpload=None, mobileOnlyDetail=None):
        super().__init__(client, None)
        self.provider = provider
        self.name = name
        self.connect_name = connectName
        self.category = category
        self.kind = kind
        self.color = color
        self.glyph = glyph
        self.icon_url = iconUrl
        self.icon_style = iconStyle
        self.attribution_required = attributionRequired
        self.connect_mode = connectMode
        self.primary_rank = primaryRank
        self.coming_soon = comingSoon
        self.deprecating = deprecating
        self.direct_upload = directUpload
        self.mobile_only_detail = mobileOnlyDetail
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'provider': repr(self.provider), f'name': repr(self.name), f'connect_name': repr(self.connect_name), f'category': repr(self.category), f'kind': repr(self.kind), f'color': repr(self.color), f'glyph': repr(self.glyph), f'icon_url': repr(self.icon_url), f'icon_style': repr(self.icon_style), f'attribution_required': repr(
            self.attribution_required), f'connect_mode': repr(self.connect_mode), f'primary_rank': repr(self.primary_rank), f'coming_soon': repr(self.coming_soon), f'deprecating': repr(self.deprecating), f'direct_upload': repr(self.direct_upload), f'mobile_only_detail': repr(self.mobile_only_detail)}
        class_name = "HealthSourceCatalogEntry"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'provider': self.provider, 'name': self.name, 'connect_name': self.connect_name, 'category': self.category, 'kind': self.kind, 'color': self.color, 'glyph': self.glyph, 'icon_url': self.icon_url, 'icon_style': self.icon_style,
                'attribution_required': self.attribution_required, 'connect_mode': self.connect_mode, 'primary_rank': self.primary_rank, 'coming_soon': self.coming_soon, 'deprecating': self.deprecating, 'direct_upload': self.direct_upload, 'mobile_only_detail': self.mobile_only_detail}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
