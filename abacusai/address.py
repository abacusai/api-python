from .return_class import AbstractApiClass


class Address(AbstractApiClass):
    """
        Address object

        Args:
            client (ApiClient): An authenticated API Client instance
            addressLine1 (str): The first line of the address
            addressLine2 (str): The second line of the address
            city (str): The city
            stateOrProvince (str): The state or province
            postalCode (str): The postal code
            country (str): The country
            additionalInfo (str): Additional information for invoice
            includeReverseCharge (bool): Whether the organization needs the reverse charge mechanism applied to invoices.
    """

    def __init__(self, client, addressLine1=None, addressLine2=None, city=None, stateOrProvince=None, postalCode=None, country=None, additionalInfo=None, includeReverseCharge=None):
        super().__init__(client, None)
        self.address_line_1 = addressLine1
        self.address_line_2 = addressLine2
        self.city = city
        self.state_or_province = stateOrProvince
        self.postal_code = postalCode
        self.country = country
        self.additional_info = additionalInfo
        self.include_reverse_charge = includeReverseCharge
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'address_line_1': repr(self.address_line_1), f'address_line_2': repr(self.address_line_2), f'city': repr(self.city), f'state_or_province': repr(self.state_or_province), f'postal_code': repr(
            self.postal_code), f'country': repr(self.country), f'additional_info': repr(self.additional_info), f'include_reverse_charge': repr(self.include_reverse_charge)}
        class_name = "Address"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'address_line_1': self.address_line_1, 'address_line_2': self.address_line_2, 'city': self.city, 'state_or_province': self.state_or_province,
                'postal_code': self.postal_code, 'country': self.country, 'additional_info': self.additional_info, 'include_reverse_charge': self.include_reverse_charge}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
