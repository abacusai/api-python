from .return_class import AbstractApiClass


class EmailDomain(AbstractApiClass):
    """
        Email domain (setup result, _getEmailDomainStatus, or list item in WebAppProject.email_domains).

        Args:
            client (ApiClient): An authenticated API Client instance
            emailDomainId (id): The email domain ID (null when NOT_CONFIGURED)
            hostname (str): The hostname
            verificationStatus (str): NOT_CONFIGURED | CONFIGURING | PENDING | VERIFIED | FAILED
            dnsRecords (list): DNS records (empty for list view)
            isAbacusManaged (bool): Whether DNS is Abacus-managed
    """

    def __init__(self, client, emailDomainId=None, hostname=None, verificationStatus=None, dnsRecords=None, isAbacusManaged=None):
        super().__init__(client, emailDomainId)
        self.email_domain_id = emailDomainId
        self.hostname = hostname
        self.verification_status = verificationStatus
        self.dns_records = dnsRecords
        self.is_abacus_managed = isAbacusManaged
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'email_domain_id': repr(self.email_domain_id), f'hostname': repr(self.hostname), f'verification_status': repr(
            self.verification_status), f'dns_records': repr(self.dns_records), f'is_abacus_managed': repr(self.is_abacus_managed)}
        class_name = "EmailDomain"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'email_domain_id': self.email_domain_id, 'hostname': self.hostname, 'verification_status':
                self.verification_status, 'dns_records': self.dns_records, 'is_abacus_managed': self.is_abacus_managed}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
