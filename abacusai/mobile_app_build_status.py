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
            expired (bool): True when a SUCCESS build is old enough that its EAS download link is likely dead
            phase (str): Native Swift builds: the archive step in progress ('archiving', 'exporting', 'uploading')
            createdAt (str): When the build was started (ISO 8601)
            testflight (dict): Native Swift builds: what Apple reported about the uploaded build
            logUrl (str): Native Swift builds: short-lived link to the archive log once the build finished
            appVersion (str): The release version the build shipped as (Android versionName, iOS CFBundleShortVersionString)
            buildNumber (int): The automatic build number of the build (Android versionCode, iOS CFBundleVersion)
    """

    def __init__(self, client, status=None, buildUrl=None, mobileAppBuildId=None, hostname=None, requiredInput=None, providers=None, selectionType=None, phoneNumbers=None, error=None, expired=None, phase=None, createdAt=None, testflight=None, logUrl=None, appVersion=None, buildNumber=None):
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
        self.expired = expired
        self.phase = phase
        self.created_at = createdAt
        self.testflight = testflight
        self.log_url = logUrl
        self.app_version = appVersion
        self.build_number = buildNumber
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'status': repr(self.status), f'build_url': repr(self.build_url), f'mobile_app_build_id': repr(self.mobile_app_build_id), f'hostname': repr(self.hostname), f'required_input': repr(self.required_input), f'providers': repr(self.providers), f'selection_type': repr(self.selection_type), f'phone_numbers': repr(
            self.phone_numbers), f'error': repr(self.error), f'expired': repr(self.expired), f'phase': repr(self.phase), f'created_at': repr(self.created_at), f'testflight': repr(self.testflight), f'log_url': repr(self.log_url), f'app_version': repr(self.app_version), f'build_number': repr(self.build_number)}
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
        resp = {'status': self.status, 'build_url': self.build_url, 'mobile_app_build_id': self.mobile_app_build_id, 'hostname': self.hostname, 'required_input': self.required_input, 'providers': self.providers, 'selection_type': self.selection_type,
                'phone_numbers': self.phone_numbers, 'error': self.error, 'expired': self.expired, 'phase': self.phase, 'created_at': self.created_at, 'testflight': self.testflight, 'log_url': self.log_url, 'app_version': self.app_version, 'build_number': self.build_number}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
