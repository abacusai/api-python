from .return_class import AbstractApiClass


class CoworkDispatchMessage(AbstractApiClass):
    """
        CoWork Dispatch Message

        Args:
            client (ApiClient): An authenticated API Client instance
            dispatchMessageId (str): The unique ID of the dispatch message
            state (str): The current state of the dispatch (e.g. pending, completed, failed, cancelled, waiting_approval)
            deploymentConversationId (id): The ID of the deployment conversation the dispatch belongs to
            deliverySeq (int): The monotonic delivery sequence number for the target desktop device
            targetDesktopDeviceId (str): The ID of the desktop device the dispatch is targeted at
            mobileDeviceId (str): The ID of the mobile device that originated the dispatch
            messageText (str): The text of the dispatched message
            workspaceOverride (str): Opaque workspace id the desktop should open for this dispatch
            llmName (str): CoWork effort tier llm_name chosen on mobile (from _listCodeBots)
            createdAt (str): The creation timestamp
            ackedAt (str): The timestamp when the dispatch was acknowledged
            completedAt (str): The timestamp when the dispatch reached a terminal state
    """

    def __init__(self, client, dispatchMessageId=None, state=None, deploymentConversationId=None, deliverySeq=None, targetDesktopDeviceId=None, mobileDeviceId=None, messageText=None, workspaceOverride=None, llmName=None, createdAt=None, ackedAt=None, completedAt=None):
        super().__init__(client, None)
        self.dispatch_message_id = dispatchMessageId
        self.state = state
        self.deployment_conversation_id = deploymentConversationId
        self.delivery_seq = deliverySeq
        self.target_desktop_device_id = targetDesktopDeviceId
        self.mobile_device_id = mobileDeviceId
        self.message_text = messageText
        self.workspace_override = workspaceOverride
        self.llm_name = llmName
        self.created_at = createdAt
        self.acked_at = ackedAt
        self.completed_at = completedAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'dispatch_message_id': repr(self.dispatch_message_id), f'state': repr(self.state), f'deployment_conversation_id': repr(self.deployment_conversation_id), f'delivery_seq': repr(self.delivery_seq), f'target_desktop_device_id': repr(self.target_desktop_device_id), f'mobile_device_id': repr(
            self.mobile_device_id), f'message_text': repr(self.message_text), f'workspace_override': repr(self.workspace_override), f'llm_name': repr(self.llm_name), f'created_at': repr(self.created_at), f'acked_at': repr(self.acked_at), f'completed_at': repr(self.completed_at)}
        class_name = "CoworkDispatchMessage"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'dispatch_message_id': self.dispatch_message_id, 'state': self.state, 'deployment_conversation_id': self.deployment_conversation_id, 'delivery_seq': self.delivery_seq, 'target_desktop_device_id': self.target_desktop_device_id,
                'mobile_device_id': self.mobile_device_id, 'message_text': self.message_text, 'workspace_override': self.workspace_override, 'llm_name': self.llm_name, 'created_at': self.created_at, 'acked_at': self.acked_at, 'completed_at': self.completed_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
