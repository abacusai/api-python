from .return_class import AbstractApiClass


class HealthDataSource(AbstractApiClass):
    """
        A connected Health data source (a wearable/device via Terra).

        Args:
            client (ApiClient): An authenticated API Client instance
            healthDataSourceId (id): The ID of the data source.
            provider (str): Provider key (oura, fitbit, garmin, ...).
            kind (str): wearable | nutrition | cgm.
            status (str): active | syncing | error | disconnected.
            errorDetail (str): Populated when status is 'error'.
            connectedAt (str): When the source was connected.
            lastSyncedAt (str): Last successful sync time.
            isLiveSync (bool): Whether this source is refreshed by pulling the aggregator, as opposed to by not refreshable at all (a retired source, kept for its history). Not derivable from `provider`: an Apple Health row is live only once the iOS SDK has connected it, so clients must branch on this rather than on a hardcoded provider list.
    """

    def __init__(self, client, healthDataSourceId=None, provider=None, kind=None, status=None, errorDetail=None, connectedAt=None, lastSyncedAt=None, isLiveSync=None):
        super().__init__(client, healthDataSourceId)
        self.health_data_source_id = healthDataSourceId
        self.provider = provider
        self.kind = kind
        self.status = status
        self.error_detail = errorDetail
        self.connected_at = connectedAt
        self.last_synced_at = lastSyncedAt
        self.is_live_sync = isLiveSync
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'health_data_source_id': repr(self.health_data_source_id), f'provider': repr(self.provider), f'kind': repr(self.kind), f'status': repr(self.status), f'error_detail': repr(
            self.error_detail), f'connected_at': repr(self.connected_at), f'last_synced_at': repr(self.last_synced_at), f'is_live_sync': repr(self.is_live_sync)}
        class_name = "HealthDataSource"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'health_data_source_id': self.health_data_source_id, 'provider': self.provider, 'kind': self.kind, 'status': self.status,
                'error_detail': self.error_detail, 'connected_at': self.connected_at, 'last_synced_at': self.last_synced_at, 'is_live_sync': self.is_live_sync}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
