from .return_class import AbstractApiClass


class DeploymentAuthToken(AbstractApiClass):
    """
        A deployment authentication token that is used to authenticate prediction requests

        Args:
            client (ApiClient): An authenticated API Client instance
            deploymentToken (str): The unique token used to authenticate requests.
            createdAt (str): The date and time when the token was created, in ISO-8601 format.
            name (str): The name associated with the authentication token.
    """

    def __init__(self, client, deploymentToken=None, createdAt=None, name=None):
        super().__init__(client, None)
        self.deployment_token = deploymentToken
        self.created_at = createdAt
        self.name = name

    def __repr__(self):
        repr_dict = {f'deployment_token': repr(self.deployment_token), f'created_at': repr(
            self.created_at), f'name': repr(self.name)}
        class_name = "DeploymentAuthToken"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'deployment_token': self.deployment_token,
                'created_at': self.created_at, 'name': self.name}
        return {key: value for key, value in resp.items() if value is not None}
