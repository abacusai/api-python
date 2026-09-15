from .return_class import AbstractApiClass


class MobileBuildInfo(AbstractApiClass):
    """
        Mobile app build information for a specific build type.

        Args:
            client (ApiClient): An authenticated API Client instance
            type (str): The type of mobile app build (APK, AAB, IOS).
            buildUrl (str): The URL to download the mobile app build.
            status (str): The status of the mobile app build.
            llmArtifactId (str): The artifact id associated with the build.
            mobileAppBuildId (str): The mobile app build id.
            expired (bool): True when a SUCCESS build is old enough that its EAS download link is likely dead.
            appVersion (str): The release version the build shipped as (Android versionName, iOS CFBundleShortVersionString)
            buildNumber (int): The automatic build number of the build (Android versionCode, iOS CFBundleVersion)
    """

    def __init__(self, client, type=None, buildUrl=None, status=None, llmArtifactId=None, mobileAppBuildId=None, expired=None, appVersion=None, buildNumber=None):
        super().__init__(client, None)
        self.type = type
        self.build_url = buildUrl
        self.status = status
        self.llm_artifact_id = llmArtifactId
        self.mobile_app_build_id = mobileAppBuildId
        self.expired = expired
        self.app_version = appVersion
        self.build_number = buildNumber
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'type': repr(self.type), f'build_url': repr(self.build_url), f'status': repr(self.status), f'llm_artifact_id': repr(self.llm_artifact_id), f'mobile_app_build_id': repr(
            self.mobile_app_build_id), f'expired': repr(self.expired), f'app_version': repr(self.app_version), f'build_number': repr(self.build_number)}
        class_name = "MobileBuildInfo"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'type': self.type, 'build_url': self.build_url, 'status': self.status, 'llm_artifact_id': self.llm_artifact_id,
                'mobile_app_build_id': self.mobile_app_build_id, 'expired': self.expired, 'app_version': self.app_version, 'build_number': self.build_number}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
