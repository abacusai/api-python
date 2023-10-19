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
        repr_dict = {f'document': repr(self.document), f'score': repr(self.score), f'properties': repr(
            self.properties), f'pages': repr(self.pages), f'bounding_boxes': repr(self.bounding_boxes)}
        class_name = "DocumentRetrieverLookupResult"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'document': self.document, 'score': self.score, 'properties': self.properties,
                'pages': self.pages, 'bounding_boxes': self.bounding_boxes}
        return {key: value for key, value in resp.items() if value is not None}
