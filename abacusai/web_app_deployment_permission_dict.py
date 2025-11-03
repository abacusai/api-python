from .return_class import AbstractApiClass


class WebAppDeploymentPermissionDict(AbstractApiClass):
    """
        Web app deployment permission dict.

        Args:
            client (ApiClient): An authenticated API Client instance
            deploymentPermissions (dict): Dictionary containing deployment ID as key and list of tuples containing (user_group_id, permission) as value.
    """

    def __init__(self, client, deploymentPermissions=None):
        super().__init__(client, None)
        self.deployment_permissions = deploymentPermissions
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'deployment_permissions': repr(
            self.deployment_permissions)}
        class_name = "WebAppDeploymentPermissionDict"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'deployment_permissions': self.deployment_permissions}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
