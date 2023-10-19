from .return_class import AbstractApiClass


class DocumentAnnotation(AbstractApiClass):
    """
        An annotation for a document in a document store

        Args:
            client (ApiClient): An authenticated API Client instance
            annotation (str): The name of the annotation
            boundingBox (dict): The bounding box for this annotation
            createdAt (str): The timestamp at which the annotation was first used
            count (int): The number of this annotation used in the document store
    """

    def __init__(self, client, annotation=None, boundingBox=None, createdAt=None, count=None):
        super().__init__(client, None)
        self.annotation = annotation
        self.bounding_box = boundingBox
        self.created_at = createdAt
        self.count = count

    def __repr__(self):
        repr_dict = {f'annotation': repr(self.annotation), f'bounding_box': repr(
            self.bounding_box), f'created_at': repr(self.created_at), f'count': repr(self.count)}
        class_name = "DocumentAnnotation"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'annotation': self.annotation, 'bounding_box': self.bounding_box,
                'created_at': self.created_at, 'count': self.count}
        return {key: value for key, value in resp.items() if value is not None}
