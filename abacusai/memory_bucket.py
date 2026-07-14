from .return_class import AbstractApiClass


class MemoryBucket(AbstractApiClass):
    """
        A consolidated long-term memory bucket for the user (the DeepAgent "dreaming" brain).

        Args:
            client (ApiClient): An authenticated API Client instance
            bucketKey (str): Stable slug for the bucket (also the rendered filename stem).
            name (str): Human-readable bucket title.
            shortDescription (str): One-line description of the bucket.
            priority (int): Importance, 1 (low) to 5 (high).
            content (str): Markdown body of the bucket.
            updatedAtTimestamp (int): Last-updated time (unix seconds).
            humanEdited (bool): Whether the user has edited this bucket (the cron is additive-only for it).
    """

    def __init__(self, client, bucketKey=None, name=None, shortDescription=None, priority=None, content=None, updatedAtTimestamp=None, humanEdited=None):
        super().__init__(client, None)
        self.bucket_key = bucketKey
        self.name = name
        self.short_description = shortDescription
        self.priority = priority
        self.content = content
        self.updated_at_timestamp = updatedAtTimestamp
        self.human_edited = humanEdited
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'bucket_key': repr(self.bucket_key), f'name': repr(self.name), f'short_description': repr(self.short_description), f'priority': repr(
            self.priority), f'content': repr(self.content), f'updated_at_timestamp': repr(self.updated_at_timestamp), f'human_edited': repr(self.human_edited)}
        class_name = "MemoryBucket"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'bucket_key': self.bucket_key, 'name': self.name, 'short_description': self.short_description, 'priority': self.priority,
                'content': self.content, 'updated_at_timestamp': self.updated_at_timestamp, 'human_edited': self.human_edited}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
