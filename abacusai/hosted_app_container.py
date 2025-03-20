from .return_class import AbstractApiClass


class HostedAppContainer(AbstractApiClass):
    """
        Hosted app container information.

        Args:
            client (ApiClient): An authenticated API Client instance
            hostedAppContainerId (id): The ID of the hosted app container
            hostedAppId (id): The ID of the hosted app
            deploymentConversationId (id): The deployment conversation ID
            hostedAppVersion (id): The instance of the hosted app
            name (str): The name of the hosted app
            createdAt (str): Creation timestamp
            updatedAt (str): Last update timestamp
            containerImage (str): Container image name
            route (str): Container route
            appConfig (dict): App configuration
            isDev (bool): Whether this is a dev container
            lifecycle (str): Container lifecycle status (PENDING/DEPLOYING/ACTIVE/FAILED/STOPPED/DELETING)
            status (str): Container status (RUNNING/STOPPED/DEPLOYING/FAILED)
    """

    def __init__(self, client, hostedAppContainerId=None, hostedAppId=None, deploymentConversationId=None, hostedAppVersion=None, name=None, createdAt=None, updatedAt=None, containerImage=None, route=None, appConfig=None, isDev=None, lifecycle=None, status=None):
        super().__init__(client, hostedAppContainerId)
        self.hosted_app_container_id = hostedAppContainerId
        self.hosted_app_id = hostedAppId
        self.deployment_conversation_id = deploymentConversationId
        self.hosted_app_version = hostedAppVersion
        self.name = name
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.container_image = containerImage
        self.route = route
        self.app_config = appConfig
        self.is_dev = isDev
        self.lifecycle = lifecycle
        self.status = status
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'hosted_app_container_id': repr(self.hosted_app_container_id), f'hosted_app_id': repr(self.hosted_app_id), f'deployment_conversation_id': repr(self.deployment_conversation_id), f'hosted_app_version': repr(self.hosted_app_version), f'name': repr(self.name), f'created_at': repr(
            self.created_at), f'updated_at': repr(self.updated_at), f'container_image': repr(self.container_image), f'route': repr(self.route), f'app_config': repr(self.app_config), f'is_dev': repr(self.is_dev), f'lifecycle': repr(self.lifecycle), f'status': repr(self.status)}
        class_name = "HostedAppContainer"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'hosted_app_container_id': self.hosted_app_container_id, 'hosted_app_id': self.hosted_app_id, 'deployment_conversation_id': self.deployment_conversation_id, 'hosted_app_version': self.hosted_app_version, 'name': self.name,
                'created_at': self.created_at, 'updated_at': self.updated_at, 'container_image': self.container_image, 'route': self.route, 'app_config': self.app_config, 'is_dev': self.is_dev, 'lifecycle': self.lifecycle, 'status': self.status}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
