from .return_class import AbstractApiClass


class Annotation(AbstractApiClass):
    """
        An Annotation Store Annotation

        Args:
            client (ApiClient): An authenticated API Client instance
            annotationType (str): A name determining the type of annotation and how to interpret the annotation value data, e.g. as a label, bounding box, etc.
            annotationValue (dict): JSON-compatible value of the annotation. The format of the value is determined by the annotation type.
            comments (dict): Comments about the annotation. This is a dictionary of feature name to the corresponding comment.
            metadata (dict): Metadata about the annotation.
    """

    def __init__(self, client, annotationType=None, annotationValue=None, comments=None, metadata=None):
        super().__init__(client, None)
        self.annotation_type = annotationType
        self.annotation_value = annotationValue
        self.comments = comments
        self.metadata = metadata

    def __repr__(self):
        repr_dict = {f'annotation_type': repr(self.annotation_type), f'annotation_value': repr(
            self.annotation_value), f'comments': repr(self.comments), f'metadata': repr(self.metadata)}
        class_name = "Annotation"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'annotation_type': self.annotation_type, 'annotation_value': self.annotation_value,
                'comments': self.comments, 'metadata': self.metadata}
        return {key: value for key, value in resp.items() if value is not None}
