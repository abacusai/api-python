from .return_class import AbstractApiClass


class DeploymentConversationExport(AbstractApiClass):
    """
        A deployment conversation html export, to be used for downloading the conversation.

        Args:
            client (ApiClient): An authenticated API Client instance
            deploymentConversationId (str): The unique identifier of the deployment conversation.
            conversationExportHtml (str): The html string of the deployment conversation.
    """

    def __init__(self, client, deploymentConversationId=None, conversationExportHtml=None):
        super().__init__(client, None)
        self.deployment_conversation_id = deploymentConversationId
        self.conversation_export_html = conversationExportHtml

    def __repr__(self):
        repr_dict = {f'deployment_conversation_id': repr(
            self.deployment_conversation_id), f'conversation_export_html': repr(self.conversation_export_html)}
        class_name = "DeploymentConversationExport"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'deployment_conversation_id': self.deployment_conversation_id,
                'conversation_export_html': self.conversation_export_html}
        return {key: value for key, value in resp.items() if value is not None}
