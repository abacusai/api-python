from .return_class import AbstractApiClass


class ExternalApplication(AbstractApiClass):
    """
        An external application.

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name of the external application.
            externalApplicationId (str): The unique identifier of the external application.
            deploymentId (str): The deployment id associated with the external application.
            description (str): The description of the external application.
            logo (str): The logo.
            theme (dict): The theme used for the External Application.
            userGroupIds (list): A list of App User Groups with access to this external application
            useCase (str): Use Case of the project of this deployment
            isAgent (bool): Whether the external application is an agent.
            status (str): The status of the deployment.
            deploymentConversationRetentionHours (int): The retention policy for the external application.
            managedUserService (str): The external service that is managing the user accounts.
            predictionOverrides (dict): The prediction overrides for the external application.
            isSystemCreated (bool): Whether the external application is system created.
            isCustomizable (bool): Whether the external application is customizable.
            isDeprecated (bool): Whether the external application is deprecated. Only applicable for system created bots. Deprecated external applications will not show in the UI.
            isVisible (bool): Whether the external application should be shown in the dropdown.
            hasThinkingOption (bool): Whether to show the thinking option in the toolbar.
            onlyImageGenEnabled (bool): Whether to LLM only allows image generation.
    """

    def __init__(self, client, name=None, externalApplicationId=None, deploymentId=None, description=None, logo=None, theme=None, userGroupIds=None, useCase=None, isAgent=None, status=None, deploymentConversationRetentionHours=None, managedUserService=None, predictionOverrides=None, isSystemCreated=None, isCustomizable=None, isDeprecated=None, isVisible=None, hasThinkingOption=None, onlyImageGenEnabled=None):
        super().__init__(client, externalApplicationId)
        self.name = name
        self.external_application_id = externalApplicationId
        self.deployment_id = deploymentId
        self.description = description
        self.logo = logo
        self.theme = theme
        self.user_group_ids = userGroupIds
        self.use_case = useCase
        self.is_agent = isAgent
        self.status = status
        self.deployment_conversation_retention_hours = deploymentConversationRetentionHours
        self.managed_user_service = managedUserService
        self.prediction_overrides = predictionOverrides
        self.is_system_created = isSystemCreated
        self.is_customizable = isCustomizable
        self.is_deprecated = isDeprecated
        self.is_visible = isVisible
        self.has_thinking_option = hasThinkingOption
        self.only_image_gen_enabled = onlyImageGenEnabled
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'external_application_id': repr(self.external_application_id), f'deployment_id': repr(self.deployment_id), f'description': repr(self.description), f'logo': repr(self.logo), f'theme': repr(self.theme), f'user_group_ids': repr(self.user_group_ids), f'use_case': repr(self.use_case), f'is_agent': repr(self.is_agent), f'status': repr(self.status), f'deployment_conversation_retention_hours': repr(
            self.deployment_conversation_retention_hours), f'managed_user_service': repr(self.managed_user_service), f'prediction_overrides': repr(self.prediction_overrides), f'is_system_created': repr(self.is_system_created), f'is_customizable': repr(self.is_customizable), f'is_deprecated': repr(self.is_deprecated), f'is_visible': repr(self.is_visible), f'has_thinking_option': repr(self.has_thinking_option), f'only_image_gen_enabled': repr(self.only_image_gen_enabled)}
        class_name = "ExternalApplication"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'external_application_id': self.external_application_id, 'deployment_id': self.deployment_id, 'description': self.description, 'logo': self.logo, 'theme': self.theme, 'user_group_ids': self.user_group_ids, 'use_case': self.use_case, 'is_agent': self.is_agent, 'status': self.status, 'deployment_conversation_retention_hours': self.deployment_conversation_retention_hours,
                'managed_user_service': self.managed_user_service, 'prediction_overrides': self.prediction_overrides, 'is_system_created': self.is_system_created, 'is_customizable': self.is_customizable, 'is_deprecated': self.is_deprecated, 'is_visible': self.is_visible, 'has_thinking_option': self.has_thinking_option, 'only_image_gen_enabled': self.only_image_gen_enabled}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def update(self, name: str = None, description: str = None, theme: dict = None, deployment_id: str = None, deployment_conversation_retention_hours: int = None, reset_retention_policy: bool = False):
        """
        Updates an External Application.

        Args:
            name (str): The name of the External Application.
            description (str): The description of the External Application. This will be shown to users when they access the External Application.
            theme (dict): The visual theme of the External Application.
            deployment_id (str): The ID of the deployment to use.
            deployment_conversation_retention_hours (int): The number of hours to retain the conversations for.
            reset_retention_policy (bool): If true, the retention policy will be removed.

        Returns:
            ExternalApplication: The updated External Application.
        """
        return self.client.update_external_application(self.external_application_id, name, description, theme, deployment_id, deployment_conversation_retention_hours, reset_retention_policy)

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
