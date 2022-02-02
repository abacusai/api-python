from .return_class import AbstractApiClass


class UseCase(AbstractApiClass):
    """
        A Project Use Case

        Args:
            client (ApiClient): An authenticated API Client instance
            useCase (str): The enum value for this use case
            prettyName (str): A user-friendly name
            description (str): A description for this use case
    """

    def __init__(self, client, useCase=None, prettyName=None, description=None):
        super().__init__(client, None)
        self.use_case = useCase
        self.pretty_name = prettyName
        self.description = description

    def __repr__(self):
        return f"UseCase(use_case={repr(self.use_case)},\n  pretty_name={repr(self.pretty_name)},\n  description={repr(self.description)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'use_case': self.use_case, 'pretty_name': self.pretty_name, 'description': self.description}
