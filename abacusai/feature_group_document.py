from .return_class import AbstractApiClass


class FeatureGroupDocument(AbstractApiClass):
    """
        A document of a feature group.

        Args:
            client (ApiClient): An authenticated API Client instance
            featureGroupId (str): The ID of the feature group this row belongs to.
            docId (str): Unique document id
            status (str): The status of the document processing
    """

    def __init__(self, client, featureGroupId=None, docId=None, status=None):
        super().__init__(client, None)
        self.feature_group_id = featureGroupId
        self.doc_id = docId
        self.status = status
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'feature_group_id': repr(self.feature_group_id), f'doc_id': repr(
            self.doc_id), f'status': repr(self.status)}
        class_name = "FeatureGroupDocument"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'feature_group_id': self.feature_group_id,
                'doc_id': self.doc_id, 'status': self.status}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
