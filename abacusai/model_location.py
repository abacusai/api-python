from .return_class import AbstractApiClass


class ModelLocation(AbstractApiClass):
    """
        Location information for plug and play model.

        Args:
            client (ApiClient): An authenticated API Client instance
            location (str): Location of the plug and play model.
            artifactNames (dict): Representations of the names of the artifacts used to create the model.
    """

    def __init__(self, client, location=None, artifactNames=None):
        super().__init__(client, None)
        self.location = location
        self.artifact_names = artifactNames

    def __repr__(self):
        return f"ModelLocation(location={repr(self.location)},\n  artifact_names={repr(self.artifact_names)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'location': self.location, 'artifact_names': self.artifact_names}
