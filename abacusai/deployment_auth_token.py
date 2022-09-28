from .return_class import AbstractApiClass


class DeploymentAuthToken(AbstractApiClass):
    """
        A deployment authentication token that is used to authenticate prediction requests

        Args:
            client (ApiClient): An authenticated API Client instance
            deploymentToken (str): The unique token used to authenticate requests
            createdAt (str): When the token was created
            name (str): 
    """

    def __init__(self, client, deploymentToken=None, createdAt=None, name=None):
        super().__init__(client, None)
        self.deployment_token = deploymentToken
        self.created_at = createdAt
        self.name = name

    def __repr__(self):
        return f"DeploymentAuthToken(deployment_token={repr(self.deployment_token)},\n  created_at={repr(self.created_at)},\n  name={repr(self.name)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'deployment_token': self.deployment_token, 'created_at': self.created_at, 'name': self.name}
