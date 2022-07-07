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
        return f"DocumentAnnotation(annotation={repr(self.annotation)},\n  bounding_box={repr(self.bounding_box)},\n  created_at={repr(self.created_at)},\n  count={repr(self.count)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'annotation': self.annotation, 'bounding_box': self.bounding_box, 'created_at': self.created_at, 'count': self.count}
