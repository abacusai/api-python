from .return_class import AbstractApiClass


class AvatarVoiceCandidate(AbstractApiClass):
    """
        Avatar Voice Candidate — one designed voice sample from generateAvatarVoice.

        Args:
            client (ApiClient): An authenticated API Client instance
            s3Key (str): Opaque sample reference — pass back to attachAvatarVoice as generatedS3Key
            previewUrl (str): Signed URL of the sample, for auditioning it before it is kept
    """

    def __init__(self, client, s3Key=None, previewUrl=None):
        super().__init__(client, None)
        self.s3_key = s3Key
        self.preview_url = previewUrl
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f's3_key': repr(self.s3_key),
                     f'preview_url': repr(self.preview_url)}
        class_name = "AvatarVoiceCandidate"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'s3_key': self.s3_key, 'preview_url': self.preview_url}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
