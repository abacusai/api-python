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

    def __repr__(self):
        return f"FeatureGroupDocument(feature_group_id={repr(self.feature_group_id)},\n  doc_id={repr(self.doc_id)},\n  status={repr(self.status)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'feature_group_id': self.feature_group_id, 'doc_id': self.doc_id, 'status': self.status}
