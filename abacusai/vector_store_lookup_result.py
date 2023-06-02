from .return_class import AbstractApiClass


class VectorStoreLookupResult(AbstractApiClass):
    """
        Result of a vector store lookup.

        Args:
            client (ApiClient): An authenticated API Client instance
            document (str): The document that was looked up.
            score (float): Score of the document with respect to the query.
    """

    def __init__(self, client, document=None, score=None):
        super().__init__(client, None)
        self.document = document
        self.score = score

    def __repr__(self):
        return f"VectorStoreLookupResult(document={repr(self.document)},\n  score={repr(self.score)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'document': self.document, 'score': self.score}
