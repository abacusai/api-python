from .return_class import AbstractApiClass


class MobileAppBuildStatus(AbstractApiClass):
    """
        Status and details of a mobile app build.

        Args:
            client (ApiClient): An authenticated API Client instance
            status (str): Current build status ('PENDING', 'SUCCESS', 'FAILED', 'CANCELLED')
            buildUrl (str): URL to download the built artifact when SUCCESS
            mobileAppBuildId (str): build identifier
            hostname (str): The hostname associated with the build
            requiredInput (str): The required input for the build
            providers (list): Apple provider/team options when awaiting a selection
            selectionType (str): Whether the pending selection is for a 'team' or a 'provider'
            phoneNumbers (list): Trusted phone number options when awaiting a 2FA phone selection
            error (str): The error message for the build
    """

    def __init__(self, client, status=None, buildUrl=None, mobileAppBuildId=None, hostname=None, requiredInput=None, providers=None, selectionType=None, phoneNumbers=None, error=None):
        super().__init__(client, None)
        self.status = status
        self.build_url = buildUrl
        self.mobile_app_build_id = mobileAppBuildId
        self.hostname = hostname
        self.required_input = requiredInput
        self.providers = providers
        self.selection_type = selectionType
        self.phone_numbers = phoneNumbers
        self.error = error
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'status': repr(self.status), f'build_url': repr(self.build_url), f'mobile_app_build_id': repr(self.mobile_app_build_id), f'hostname': repr(self.hostname), f'required_input': repr(
            self.required_input), f'providers': repr(self.providers), f'selection_type': repr(self.selection_type), f'phone_numbers': repr(self.phone_numbers), f'error': repr(self.error)}
        class_name = "MobileAppBuildStatus"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'status': self.status, 'build_url': self.build_url, 'mobile_app_build_id': self.mobile_app_build_id, 'hostname': self.hostname,
                'required_input': self.required_input, 'providers': self.providers, 'selection_type': self.selection_type, 'phone_numbers': self.phone_numbers, 'error': self.error}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
