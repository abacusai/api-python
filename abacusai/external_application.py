from .return_class import AbstractApiClass


class ExternalApplication(AbstractApiClass):
    """
        An external application.

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name of the external application.
            externalApplicationId (str): The unique identifier of the external application.
            deploymentId (str): The deployment id associated with the external application.
            logo (str): The logo.
            theme (dict): The theme used for the External Application.
            userGroupIds (list): A list of App User Groups with access to this external application
            useCase (str): Use Case of the project of this deployment
    """

    def __init__(self, client, name=None, externalApplicationId=None, deploymentId=None, logo=None, theme=None, userGroupIds=None, useCase=None):
        super().__init__(client, externalApplicationId)
        self.name = name
        self.external_application_id = externalApplicationId
        self.deployment_id = deploymentId
        self.logo = logo
        self.theme = theme
        self.user_group_ids = userGroupIds
        self.use_case = useCase

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'external_application_id': repr(self.external_application_id), f'deployment_id': repr(
            self.deployment_id), f'logo': repr(self.logo), f'theme': repr(self.theme), f'user_group_ids': repr(self.user_group_ids), f'use_case': repr(self.use_case)}
        class_name = "ExternalApplication"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'external_application_id': self.external_application_id, 'deployment_id': self.deployment_id,
                'logo': self.logo, 'theme': self.theme, 'user_group_ids': self.user_group_ids, 'use_case': self.use_case}
        return {key: value for key, value in resp.items() if value is not None}

    def update(self, name: str = None, theme: dict = None):
        """
        Updates an External Application.

        Args:
            name (str): The name of the External Application.
            theme (dict): The visual theme of the External Application.

        Returns:
            ExternalApplication: The updated External Application.
        """
        return self.client.update_external_application(self.external_application_id, name, theme)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            ExternalApplication: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Describes an External Application.

        Args:
            external_application_id (str): The ID of the External Application.

        Returns:
            ExternalApplication: The External Application.
        """
        return self.client.describe_external_application(self.external_application_id)

    def delete(self):
        """
        Deletes an External Application.

        Args:
            external_application_id (str): The ID of the External Application.
        """
        return self.client.delete_external_application(self.external_application_id)
