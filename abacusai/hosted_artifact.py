from .return_class import AbstractApiClass


class HostedArtifact(AbstractApiClass):
    """
        A hosted artifact being served by the platform.

        Args:
            client (ApiClient): An authenticated API Client instance
            hostnames (list): The urls at which the application is being hosted.
            artifactType (str): The type of artifact being hosted.
            llmArtifactId (str): The artifact id being hosted.
            lifecycle (str): The lifecycle of the artifact.
            externalApplicationId (str): Agent that deployed this application.
            deploymentConversationId (str): Conversation that created deployed this artifact, null if not applicable.
            conversationSequenceNumber (number(integer)): Conversation event associated with this artifact, null if not applicable.
            accessLevel (str): The access level of the hosted artifact (PUBLIC, PRIVATE, OWNER_ONLY, DEDICATED).
            isThrottled (bool): Whether the artifact has been temporarily suspended due to high resource usage.
    """

    def __init__(self, client, hostnames=None, artifactType=None, llmArtifactId=None, lifecycle=None, externalApplicationId=None, deploymentConversationId=None, conversationSequenceNumber=None, accessLevel=None, isThrottled=None):
        super().__init__(client, None)
        self.hostnames = hostnames
        self.artifact_type = artifactType
        self.llm_artifact_id = llmArtifactId
        self.lifecycle = lifecycle
        self.external_application_id = externalApplicationId
        self.deployment_conversation_id = deploymentConversationId
        self.conversation_sequence_number = conversationSequenceNumber
        self.access_level = accessLevel
        self.is_throttled = isThrottled
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'hostnames': repr(self.hostnames), f'artifact_type': repr(self.artifact_type), f'llm_artifact_id': repr(self.llm_artifact_id), f'lifecycle': repr(self.lifecycle), f'external_application_id': repr(
            self.external_application_id), f'deployment_conversation_id': repr(self.deployment_conversation_id), f'conversation_sequence_number': repr(self.conversation_sequence_number), f'access_level': repr(self.access_level), f'is_throttled': repr(self.is_throttled)}
        class_name = "HostedArtifact"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'hostnames': self.hostnames, 'artifact_type': self.artifact_type, 'llm_artifact_id': self.llm_artifact_id, 'lifecycle': self.lifecycle, 'external_application_id': self.external_application_id,
                'deployment_conversation_id': self.deployment_conversation_id, 'conversation_sequence_number': self.conversation_sequence_number, 'access_level': self.access_level, 'is_throttled': self.is_throttled}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
