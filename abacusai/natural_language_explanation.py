from .return_class import AbstractApiClass


class NaturalLanguageExplanation(AbstractApiClass):
    """
        Natural language explanation of an artifact/object

        Args:
            client (ApiClient): An authenticated API Client instance
            shortExplanation (str): succinct explanation of the artifact
            longExplanation (str): Longer and verbose explanation of the artifact
            isOutdated (bool): Flag indicating whether the explanation is outdated due to a change in the underlying artifact
            htmlExplanation (str): HTML formatted explanation of the artifact
    """

    def __init__(self, client, shortExplanation=None, longExplanation=None, isOutdated=None, htmlExplanation=None):
        super().__init__(client, None)
        self.short_explanation = shortExplanation
        self.long_explanation = longExplanation
        self.is_outdated = isOutdated
        self.html_explanation = htmlExplanation

    def __repr__(self):
        return f"NaturalLanguageExplanation(short_explanation={repr(self.short_explanation)},\n  long_explanation={repr(self.long_explanation)},\n  is_outdated={repr(self.is_outdated)},\n  html_explanation={repr(self.html_explanation)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'short_explanation': self.short_explanation, 'long_explanation': self.long_explanation, 'is_outdated': self.is_outdated, 'html_explanation': self.html_explanation}
