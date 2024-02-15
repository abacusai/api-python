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
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'short_explanation': repr(self.short_explanation), f'long_explanation': repr(
            self.long_explanation), f'is_outdated': repr(self.is_outdated), f'html_explanation': repr(self.html_explanation)}
        class_name = "NaturalLanguageExplanation"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'short_explanation': self.short_explanation, 'long_explanation': self.long_explanation,
                'is_outdated': self.is_outdated, 'html_explanation': self.html_explanation}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
