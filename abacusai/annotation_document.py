from .return_class import AbstractApiClass


class AnnotationDocument(AbstractApiClass):
    """
        Document to be annotated.

        Args:
            client (ApiClient): An authenticated API Client instance
            docId (str): The docstore Document ID of the document.
            featureGroupRowIdentifier (str): The key value of the feature group row the annotation is on. Usually the primary key value.
            featureGroupRowIndex (int): The index of the document row in the feature group.
            totalRows (int): The total number of rows in the feature group.
            isAnnotationPresent (bool): Whether the document already has an annotation. Returns None if feature group is not under annotations review mode.
    """

    def __init__(self, client, docId=None, featureGroupRowIdentifier=None, featureGroupRowIndex=None, totalRows=None, isAnnotationPresent=None):
        super().__init__(client, None)
        self.doc_id = docId
        self.feature_group_row_identifier = featureGroupRowIdentifier
        self.feature_group_row_index = featureGroupRowIndex
        self.total_rows = totalRows
        self.is_annotation_present = isAnnotationPresent

    def __repr__(self):
        repr_dict = {f'doc_id': repr(self.doc_id), f'feature_group_row_identifier': repr(self.feature_group_row_identifier), f'feature_group_row_index': repr(
            self.feature_group_row_index), f'total_rows': repr(self.total_rows), f'is_annotation_present': repr(self.is_annotation_present)}
        class_name = "AnnotationDocument"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'doc_id': self.doc_id, 'feature_group_row_identifier': self.feature_group_row_identifier,
                'feature_group_row_index': self.feature_group_row_index, 'total_rows': self.total_rows, 'is_annotation_present': self.is_annotation_present}
        return {key: value for key, value in resp.items() if value is not None}
