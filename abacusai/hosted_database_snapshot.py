from .return_class import AbstractApiClass


class HostedDatabaseSnapshot(AbstractApiClass):
    """
        Hosted Database Snapshot

        Args:
            client (ApiClient): An authenticated API Client instance
            hostedDatabaseSnapshotId (id): The ID of the hosted database snapshot
            srcHostedDatabaseId (id): The ID of the source hosted database
            createdAt (str): The creation timestamp
            updatedAt (str): The last update timestamp
            lifecycle (str): The lifecycle of the hosted database snapshot
    """

    def __init__(self, client, hostedDatabaseSnapshotId=None, srcHostedDatabaseId=None, createdAt=None, updatedAt=None, lifecycle=None):
        super().__init__(client, hostedDatabaseSnapshotId)
        self.hosted_database_snapshot_id = hostedDatabaseSnapshotId
        self.src_hosted_database_id = srcHostedDatabaseId
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.lifecycle = lifecycle
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'hosted_database_snapshot_id': repr(self.hosted_database_snapshot_id), f'src_hosted_database_id': repr(
            self.src_hosted_database_id), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at), f'lifecycle': repr(self.lifecycle)}
        class_name = "HostedDatabaseSnapshot"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'hosted_database_snapshot_id': self.hosted_database_snapshot_id, 'src_hosted_database_id': self.src_hosted_database_id,
                'created_at': self.created_at, 'updated_at': self.updated_at, 'lifecycle': self.lifecycle}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
