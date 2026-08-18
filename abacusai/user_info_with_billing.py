from .return_class import AbstractApiClass


class UserInfoWithBilling(AbstractApiClass):
    """
        The billing-funnel page-load bundle: the payloads of _getUserInfo, _getBillingInfo,

        Args:
            client (ApiClient): An authenticated API Client instance
            tiersInfo (list): The _getTiersInfo payload.
            publicKey (str): The _getPublicKey payload (Stripe publishable key).
            userInfo (InternalUserInfo): The _getUserInfo payload.
            billingInfo (BillingInfo): The _getBillingInfo payload.
            activePromotion (ActivePromo): The getActivePromotion payload.
    """

    def __init__(self, client, tiersInfo=None, publicKey=None, userInfo={}, billingInfo={}, activePromotion={}):
        super().__init__(client, None)
        self.tiers_info = tiersInfo
        self.public_key = publicKey
        self.user_info = client._build_class(InternalUserInfo, userInfo)
        self.billing_info = client._build_class(BillingInfo, billingInfo)
        self.active_promotion = client._build_class(
            ActivePromo, activePromotion)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'tiers_info': repr(self.tiers_info), f'public_key': repr(self.public_key), f'user_info': repr(
            self.user_info), f'billing_info': repr(self.billing_info), f'active_promotion': repr(self.active_promotion)}
        class_name = "UserInfoWithBilling"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'tiers_info': self.tiers_info, 'public_key': self.public_key, 'user_info': self._get_attribute_as_dict(
            self.user_info), 'billing_info': self._get_attribute_as_dict(self.billing_info), 'active_promotion': self._get_attribute_as_dict(self.active_promotion)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
