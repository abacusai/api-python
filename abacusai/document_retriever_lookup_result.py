from .return_class import AbstractApiClass


class DocumentRetrieverLookupResult(AbstractApiClass):
    """
        Result of a document retriever lookup.

        Args:
            client (ApiClient): An authenticated API Client instance
            document (str): The document that was looked up.
            score (float): Score of the document with respect to the query.
            properties (dict): Properties of the retrieved documents.
            pages (list): Pages of the retrieved text from the original document.
            boundingBoxes (list): Bounding boxes of the retrieved text from the original document.
    """

    def __init__(self, client, document=None, score=None, properties=None, pages=None, boundingBoxes=None):
        super().__init__(client, None)
        self.document = document
        self.score = score
        self.properties = properties
        self.pages = pages
        self.bounding_boxes = boundingBoxes

    def __repr__(self):
        return f"DocumentRetrieverLookupResult(document={repr(self.document)},\n  score={repr(self.score)},\n  properties={repr(self.properties)},\n  pages={repr(self.pages)},\n  bounding_boxes={repr(self.bounding_boxes)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'document': self.document, 'score': self.score, 'properties': self.properties, 'pages': self.pages, 'bounding_boxes': self.bounding_boxes}
