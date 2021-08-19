

class DeploymentAuthToken():
    '''
        A deployment authentication token that is used to authenticate prediction requests
    '''

    def __init__(self, client, deploymentToken=None, createdAt=None):
        self.client = client
        self.id = None
        self.deployment_token = deploymentToken
        self.created_at = createdAt

    def __repr__(self):
        return f"DeploymentAuthToken(deployment_token={repr(self.deployment_token)}, created_at={repr(self.created_at)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'deployment_token': self.deployment_token, 'created_at': self.created_at}
