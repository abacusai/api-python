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
    """

    def __init__(self, client, name=None, externalApplicationId=None, deploymentId=None, logo=None, theme=None, userGroupIds=None):
        super().__init__(client, externalApplicationId)
        self.name = name
        self.external_application_id = externalApplicationId
        self.deployment_id = deploymentId
        self.logo = logo
        self.theme = theme
        self.user_group_ids = userGroupIds

    def __repr__(self):
        return f"ExternalApplication(name={repr(self.name)},\n  external_application_id={repr(self.external_application_id)},\n  deployment_id={repr(self.deployment_id)},\n  logo={repr(self.logo)},\n  theme={repr(self.theme)},\n  user_group_ids={repr(self.user_group_ids)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'external_application_id': self.external_application_id, 'deployment_id': self.deployment_id, 'logo': self.logo, 'theme': self.theme, 'user_group_ids': self.user_group_ids}

    def update(self, name: str = None, logo: str = None, theme: dict = None):
        """
        Updates an External Application.

        Args:
            name (str): The name of the External Application.
            logo (str): The logo to be displayed.
            theme (dict): The visual theme of the External Application.

        Returns:
            ExternalApplication: The updated External Application.
        """
        return self.client.update_external_application(self.external_application_id, name, logo, theme)

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
