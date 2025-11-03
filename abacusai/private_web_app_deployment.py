from .return_class import AbstractApiClass


class PrivateWebAppDeployment(AbstractApiClass):
    """
        Private web app deployment list.

        Args:
            client (ApiClient): An authenticated API Client instance
            hostname (str): The hostname of the web app deployment.
            llmArtifactId (id): The ID of the LLM artifact.
            hostedAppVersion (id): The version of the hosted app.
            applicationType (str): The type of application.
            lifecycle (str): The lifecycle of the web app deployment.
            deploymentConversationId (id): The ID of the deployment conversation.
            conversationName (str): The name of the conversation.
            userId (id): The ID of the user who created the deployment.
            email (str): The email of the user who created the deployment.
            conversationType (str): The type of conversation.
            source (str): The source of the conversation.
            conversationCreatedAt (str): The timestamp when the conversation was created.
    """

    def __init__(self, client, hostname=None, llmArtifactId=None, hostedAppVersion=None, applicationType=None, lifecycle=None, deploymentConversationId=None, conversationName=None, userId=None, email=None, conversationType=None, source=None, conversationCreatedAt=None):
        super().__init__(client, None)
        self.hostname = hostname
        self.llm_artifact_id = llmArtifactId
        self.hosted_app_version = hostedAppVersion
        self.application_type = applicationType
        self.lifecycle = lifecycle
        self.deployment_conversation_id = deploymentConversationId
        self.conversation_name = conversationName
        self.user_id = userId
        self.email = email
        self.conversation_type = conversationType
        self.source = source
        self.conversation_created_at = conversationCreatedAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'hostname': repr(self.hostname), f'llm_artifact_id': repr(self.llm_artifact_id), f'hosted_app_version': repr(self.hosted_app_version), f'application_type': repr(self.application_type), f'lifecycle': repr(self.lifecycle), f'deployment_conversation_id': repr(
            self.deployment_conversation_id), f'conversation_name': repr(self.conversation_name), f'user_id': repr(self.user_id), f'email': repr(self.email), f'conversation_type': repr(self.conversation_type), f'source': repr(self.source), f'conversation_created_at': repr(self.conversation_created_at)}
        class_name = "PrivateWebAppDeployment"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'hostname': self.hostname, 'llm_artifact_id': self.llm_artifact_id, 'hosted_app_version': self.hosted_app_version, 'application_type': self.application_type, 'lifecycle': self.lifecycle, 'deployment_conversation_id': self.deployment_conversation_id,
                'conversation_name': self.conversation_name, 'user_id': self.user_id, 'email': self.email, 'conversation_type': self.conversation_type, 'source': self.source, 'conversation_created_at': self.conversation_created_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
