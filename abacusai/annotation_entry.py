from .annotation import Annotation
from .return_class import AbstractApiClass


class AnnotationEntry(AbstractApiClass):
    """
        An Annotation Store entry for an Annotation

        Args:
            client (ApiClient): An authenticated API Client instance
            featureGroupId (str): The ID of the feature group this annotation belongs to.
            featureName (str): name of the feature this annotation is on.
            docId (str): The ID of the primary document the annotation is on.
            featureGroupRowIdentifier (str): The key value of the feature group row the annotation is on (cast to string). Usually the primary key value.
            updatedAt (str): Most recent time the annotation entry was modified, e.g. creation or update time.
            annotationEntryMarker (str): The entry marker for the annotation.
            status (str): The status of labeling the document.
            lockedUntil (str): The time until which the document is locked for editing,  in ISO-8601 format.
            verificationInfo (dict): The verification info for the annotation.
            annotation (Annotation): json-compatible structure holding the type and value of the annotation.
    """

    def __init__(self, client, featureGroupId=None, featureName=None, docId=None, featureGroupRowIdentifier=None, updatedAt=None, annotationEntryMarker=None, status=None, lockedUntil=None, verificationInfo=None, annotation={}):
        super().__init__(client, None)
        self.feature_group_id = featureGroupId
        self.feature_name = featureName
        self.doc_id = docId
        self.feature_group_row_identifier = featureGroupRowIdentifier
        self.updated_at = updatedAt
        self.annotation_entry_marker = annotationEntryMarker
        self.status = status
        self.locked_until = lockedUntil
        self.verification_info = verificationInfo
        self.annotation = client._build_class(Annotation, annotation)

    def __repr__(self):
        return f"AnnotationEntry(feature_group_id={repr(self.feature_group_id)},\n  feature_name={repr(self.feature_name)},\n  doc_id={repr(self.doc_id)},\n  feature_group_row_identifier={repr(self.feature_group_row_identifier)},\n  updated_at={repr(self.updated_at)},\n  annotation_entry_marker={repr(self.annotation_entry_marker)},\n  status={repr(self.status)},\n  locked_until={repr(self.locked_until)},\n  verification_info={repr(self.verification_info)},\n  annotation={repr(self.annotation)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'feature_group_id': self.feature_group_id, 'feature_name': self.feature_name, 'doc_id': self.doc_id, 'feature_group_row_identifier': self.feature_group_row_identifier, 'updated_at': self.updated_at, 'annotation_entry_marker': self.annotation_entry_marker, 'status': self.status, 'locked_until': self.locked_until, 'verification_info': self.verification_info, 'annotation': self._get_attribute_as_dict(self.annotation)}
