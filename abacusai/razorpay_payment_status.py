from .return_class import AbstractApiClass


class RazorpayPaymentStatus(AbstractApiClass):
    """
        The status of a Razorpay payment

        Args:
            client (ApiClient): An authenticated API Client instance
            status (str): The status of the payment.
    """

    def __init__(self, client, status=None):
        super().__init__(client, None)
        self.status = status
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'status': repr(self.status)}
        class_name = "RazorpayPaymentStatus"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'status': self.status}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
