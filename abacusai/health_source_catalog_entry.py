from .return_class import AbstractApiClass


class HealthSourceCatalogEntry(AbstractApiClass):
    """
        One provider the Health Data Sources surface can show — what it is called, how it looks, and how (or

        Args:
            client (ApiClient): An authenticated API Client instance
            provider (str): Provider key (oura, fitbit, garmin, ...), matching HealthDataSource.provider.
            name (str): Brand name, shown as-is (never translated).
            category (str): Stable display key the client maps to a translated label (Wearable, Fitness, Smart scale, Nutrition, Medical, CGM).
            kind (str): wearable | nutrition | cgm — the ingestion-side data shape, not a user-facing label.
            color (str): Hex brand colour for the circular source badge.
            glyph (str): One or two characters drawn on that badge when no icon renders.
            iconUrl (str): A WHITE monochrome mark on transparency, to be composited over `color`. One asset works in light and dark, on web and native. Absent for a provider whose mark isn't published yet — draw `glyph` instead.
            connectMode (str): web_oauth (Terra's hosted OAuth from a browser) | mobile_app (on-device SDK only — the phone health stores). Derived from the same provider tuple the connect endpoint enforces, so a client that offers what this says can never be refused for offering the wrong thing.
            primaryRank (int): Tie-break order when several connected devices report the SAME metric and are otherwise equally good witnesses — lower wins, unranked providers sort last. Served so the insight cards and a client's metrics view attribute a reading to the same device.
            comingSoon (bool): Held back from launch — list it, but offer no way to connect it. Resolved per org from the health_coming_soon_sources flag, so web and app agree on what has shipped.
            deprecating (bool): Supported, but the provider is winding the integration down (Google Fit's REST API in favour of Health Connect) — best-effort.
            mobileOnlyDetail (str): Why a browser can't connect this one, for a web user. Present only when connect_mode is mobile_app.
    """

    def __init__(self, client, provider=None, name=None, category=None, kind=None, color=None, glyph=None, iconUrl=None, connectMode=None, primaryRank=None, comingSoon=None, deprecating=None, mobileOnlyDetail=None):
        super().__init__(client, None)
        self.provider = provider
        self.name = name
        self.category = category
        self.kind = kind
        self.color = color
        self.glyph = glyph
        self.icon_url = iconUrl
        self.connect_mode = connectMode
        self.primary_rank = primaryRank
        self.coming_soon = comingSoon
        self.deprecating = deprecating
        self.mobile_only_detail = mobileOnlyDetail
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'provider': repr(self.provider), f'name': repr(self.name), f'category': repr(self.category), f'kind': repr(self.kind), f'color': repr(self.color), f'glyph': repr(self.glyph), f'icon_url': repr(
            self.icon_url), f'connect_mode': repr(self.connect_mode), f'primary_rank': repr(self.primary_rank), f'coming_soon': repr(self.coming_soon), f'deprecating': repr(self.deprecating), f'mobile_only_detail': repr(self.mobile_only_detail)}
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
        resp = {'provider': self.provider, 'name': self.name, 'category': self.category, 'kind': self.kind, 'color': self.color, 'glyph': self.glyph, 'icon_url': self.icon_url,
                'connect_mode': self.connect_mode, 'primary_rank': self.primary_rank, 'coming_soon': self.coming_soon, 'deprecating': self.deprecating, 'mobile_only_detail': self.mobile_only_detail}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
