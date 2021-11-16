from .return_class import AbstractApiClass


class DeploymentAuthToken(AbstractApiClass):
    """
        A deployment authentication token that is used to authenticate prediction requests
    """

    def __init__(self, client, deploymentToken=None, createdAt=None):
        super().__init__(client, None)
        self.deployment_token = deploymentToken
        self.created_at = createdAt

    def __repr__(self):
        return f"DeploymentAuthToken(deployment_token={repr(self.deployment_token)},\n  created_at={repr(self.created_at)})"

    def to_dict(self):
        return {'deployment_token': self.deployment_token, 'created_at': self.created_at}
