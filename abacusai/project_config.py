from .return_class import AbstractApiClass


class ProjectConfig(AbstractApiClass):
    """
        Project-specific config for a feature group

        Args:
            client (ApiClient): An authenticated API Client instance
            type (str): Type of project config
            config (dict): Project-specific config for this feature group
    """

    def __init__(self, client, type=None, config=None):
        super().__init__(client, None)
        self.type = type
        self.config = config

    def __repr__(self):
        return f"ProjectConfig(type={repr(self.type)},\n  config={repr(self.config)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'type': self.type, 'config': self.config}
