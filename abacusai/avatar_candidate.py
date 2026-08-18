from .return_class import AbstractApiClass


class AvatarCandidate(AbstractApiClass):
    """
        Avatar Candidate — one generated portrait option from generateAvatarCandidates.

        Args:
            client (ApiClient): An authenticated API Client instance
            s3Key (str): Opaque candidate reference — pass back to createAvatar as imageRefs[{s3Key}]
            url (str): Signed preview URL for the candidate grid
            thumbUrl (str): Signed URL of a small copy, for the 2x2 chooser grid
    """

    def __init__(self, client, s3Key=None, url=None, thumbUrl=None):
        super().__init__(client, None)
        self.s3_key = s3Key
        self.url = url
        self.thumb_url = thumbUrl
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f's3_key': repr(self.s3_key), f'url': repr(
            self.url), f'thumb_url': repr(self.thumb_url)}
        class_name = "AvatarCandidate"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'s3_key': self.s3_key, 'url': self.url,
                'thumb_url': self.thumb_url}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
