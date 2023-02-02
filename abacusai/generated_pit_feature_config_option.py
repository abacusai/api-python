from .return_class import AbstractApiClass


class GeneratedPitFeatureConfigOption(AbstractApiClass):
    """
        The options to display for possible generated PIT aggregation functions

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The short name of the aggregation type.
            displayName (str): The display name of the aggregation type.
            default (bool): The default value for the option.
            description (str): The description of the aggregation type.
    """

    def __init__(self, client, name=None, displayName=None, default=None, description=None):
        super().__init__(client, None)
        self.name = name
        self.display_name = displayName
        self.default = default
        self.description = description

    def __repr__(self):
        return f"GeneratedPitFeatureConfigOption(name={repr(self.name)},\n  display_name={repr(self.display_name)},\n  default={repr(self.default)},\n  description={repr(self.description)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'display_name': self.display_name, 'default': self.default, 'description': self.description}
