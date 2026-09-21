from .return_class import AbstractApiClass


class MobileAppIdentifiers(AbstractApiClass):
    """
        A mobile app's store identifiers and whether each is still changeable.

        Args:
            client (ApiClient): An authenticated API Client instance
            packageName (str): Android applicationId, e.g. com.acme.fieldkit
            bundleId (str): iOS bundle identifier
            appScheme (str): Deep-link scheme. Derived, and not user-editable
            identifierSource (str): 'default' (platform-generated) or 'custom' (the user set it)
            packageNameLocked (bool): True once something outside our database committed to it
            packageNameLockReason (str): why it is fixed -- 'play_upload_artifact', 'preview_build' (a successful .apk), 'push_setup', 'already_chosen' (the user spent their one change), 'matches_other_platform' (the bundle id committed and the two must match), or 'lock_state_unavailable'
            packageNameLockedAt (str): When the row that caused the lock was created (ISO 8601)
            bundleIdLocked (bool): True once a successful iOS build reached App Store Connect
            bundleIdLockReason (str): why it is fixed -- 'apple_registration' (a successful iOS build reached App Store Connect), 'apple_app_record' (the ASC record exists, which happens before any build), 'already_chosen', 'matches_other_platform' (the package name committed and the two must match), or 'lock_state_unavailable'
            bundleIdLockedAt (str): When the row that caused the lock was created (ISO 8601)
            packageNameTakenOnStore (bool): Advisory: a published Play app already uses it. Absent when unknown
            bundleIdTakenOnStore (bool): Advisory: a published App Store app already uses it. Absent when unknown
    """

    def __init__(self, client, packageName=None, bundleId=None, appScheme=None, identifierSource=None, packageNameLocked=None, packageNameLockReason=None, packageNameLockedAt=None, bundleIdLocked=None, bundleIdLockReason=None, bundleIdLockedAt=None, packageNameTakenOnStore=None, bundleIdTakenOnStore=None):
        super().__init__(client, None)
        self.package_name = packageName
        self.bundle_id = bundleId
        self.app_scheme = appScheme
        self.identifier_source = identifierSource
        self.package_name_locked = packageNameLocked
        self.package_name_lock_reason = packageNameLockReason
        self.package_name_locked_at = packageNameLockedAt
        self.bundle_id_locked = bundleIdLocked
        self.bundle_id_lock_reason = bundleIdLockReason
        self.bundle_id_locked_at = bundleIdLockedAt
        self.package_name_taken_on_store = packageNameTakenOnStore
        self.bundle_id_taken_on_store = bundleIdTakenOnStore
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'package_name': repr(self.package_name), f'bundle_id': repr(self.bundle_id), f'app_scheme': repr(self.app_scheme), f'identifier_source': repr(self.identifier_source), f'package_name_locked': repr(self.package_name_locked), f'package_name_lock_reason': repr(self.package_name_lock_reason), f'package_name_locked_at': repr(
            self.package_name_locked_at), f'bundle_id_locked': repr(self.bundle_id_locked), f'bundle_id_lock_reason': repr(self.bundle_id_lock_reason), f'bundle_id_locked_at': repr(self.bundle_id_locked_at), f'package_name_taken_on_store': repr(self.package_name_taken_on_store), f'bundle_id_taken_on_store': repr(self.bundle_id_taken_on_store)}
        class_name = "MobileAppIdentifiers"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'package_name': self.package_name, 'bundle_id': self.bundle_id, 'app_scheme': self.app_scheme, 'identifier_source': self.identifier_source, 'package_name_locked': self.package_name_locked, 'package_name_lock_reason': self.package_name_lock_reason, 'package_name_locked_at': self.package_name_locked_at,
                'bundle_id_locked': self.bundle_id_locked, 'bundle_id_lock_reason': self.bundle_id_lock_reason, 'bundle_id_locked_at': self.bundle_id_locked_at, 'package_name_taken_on_store': self.package_name_taken_on_store, 'bundle_id_taken_on_store': self.bundle_id_taken_on_store}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
