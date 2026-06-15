from .return_class import AbstractApiClass


class CoworkPairingChallenge(AbstractApiClass):
    """
        CoWork Pairing Challenge

        Args:
            client (ApiClient): An authenticated API Client instance
            pairingId (str): The unique ID of the pairing challenge
            userCode (str): The short human-readable code used to confirm the pairing
            qrToken (str): The token encoded in the pairing QR code
            verificationDeepLink (str): The deep link used to verify the pairing from the mobile device
            expiresAt (str): The timestamp when the pairing challenge expires
            status (str): The current status of the pairing challenge (e.g. pending)
            desktopDeviceId (str): The ID of the desktop device requesting the pairing
            deviceName (str): The display name of the desktop device requesting the pairing
    """

    def __init__(self, client, pairingId=None, userCode=None, qrToken=None, verificationDeepLink=None, expiresAt=None, status=None, desktopDeviceId=None, deviceName=None):
        super().__init__(client, None)
        self.pairing_id = pairingId
        self.user_code = userCode
        self.qr_token = qrToken
        self.verification_deep_link = verificationDeepLink
        self.expires_at = expiresAt
        self.status = status
        self.desktop_device_id = desktopDeviceId
        self.device_name = deviceName
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'pairing_id': repr(self.pairing_id), f'user_code': repr(self.user_code), f'qr_token': repr(self.qr_token), f'verification_deep_link': repr(
            self.verification_deep_link), f'expires_at': repr(self.expires_at), f'status': repr(self.status), f'desktop_device_id': repr(self.desktop_device_id), f'device_name': repr(self.device_name)}
        class_name = "CoworkPairingChallenge"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'pairing_id': self.pairing_id, 'user_code': self.user_code, 'qr_token': self.qr_token, 'verification_deep_link': self.verification_deep_link,
                'expires_at': self.expires_at, 'status': self.status, 'desktop_device_id': self.desktop_device_id, 'device_name': self.device_name}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
