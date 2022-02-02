from .return_class import AbstractApiClass


class NlpSentimentPrediction(AbstractApiClass):
    """


        Args:
            client (ApiClient): An authenticated API Client instance
    """

    def __init__(self, client, ):
        super().__init__(client, None)

    def __repr__(self):
        return f"NlpSentimentPrediction()"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {}
