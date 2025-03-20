from .return_class import AbstractApiClass


class HostedApp(AbstractApiClass):
    """
        Hosted App

        Args:
            client (ApiClient): An authenticated API Client instance
            hostedAppId (id): The ID of the hosted app
            deploymentConversationId (id): The ID of the deployment conversation
            name (str): The name of the hosted app
            createdAt (str): The creation timestamp
    """

    def __init__(self, client, hostedAppId=None, deploymentConversationId=None, name=None, createdAt=None):
        super().__init__(client, hostedAppId)
        self.hosted_app_id = hostedAppId
        self.deployment_conversation_id = deploymentConversationId
        self.name = name
        self.created_at = createdAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'hosted_app_id': repr(self.hosted_app_id), f'deployment_conversation_id': repr(
            self.deployment_conversation_id), f'name': repr(self.name), f'created_at': repr(self.created_at)}
        class_name = "HostedApp"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'hosted_app_id': self.hosted_app_id, 'deployment_conversation_id':
                self.deployment_conversation_id, 'name': self.name, 'created_at': self.created_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
