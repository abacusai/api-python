from .return_class import AbstractApiClass


class RazorpayPaymentRequirements(AbstractApiClass):
    """
        A Razorpay customer and order

        Args:
            client (ApiClient): An authenticated API Client instance
            customerId (str): The unique identifier of the customer.
            orderId (str): The unique identifier of the order.
            invoiceId (str): The unique identifier of the invoice.
    """

    def __init__(self, client, customerId=None, orderId=None, invoiceId=None):
        super().__init__(client, None)
        self.customer_id = customerId
        self.order_id = orderId
        self.invoice_id = invoiceId
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'customer_id': repr(self.customer_id), f'order_id': repr(
            self.order_id), f'invoice_id': repr(self.invoice_id)}
        class_name = "RazorpayPaymentRequirements"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'customer_id': self.customer_id,
                'order_id': self.order_id, 'invoice_id': self.invoice_id}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
