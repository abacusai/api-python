from .return_class import AbstractApiClass


class DaemonTaskEvent(AbstractApiClass):
    """
        One delivery to an event-based daemon task (a webhook, inbox or test event that was dispatched as a run).

        Args:
            client (ApiClient): An authenticated API Client instance
            daemonTaskVersion (str): The id of the run the event dispatched.
            deploymentConversationId (str): The run conversation.
            lifecycle (str): The run lifecycle (PENDING, EXECUTING, COMPLETED, FAILED, WAITING_FOR_USER_INPUT).
            failureReason (str): Why the run failed, when it did.
            isLimitBlocked (bool): Whether the run was skipped by a credit/plan limit rather than failing.
            isTestRun (bool): Whether the user sent the event as a test.
            sequenceNumber (int): 0-based run number.
            createdAt (str): When the event arrived.
            updatedAt (str): When the run last changed lifecycle.
            source (str): Where the event came from (github, slack, gmail, webhook, ...).
            eventType (str): The source's event type (pull_request, email_received, test, ...).
            payload (dict): The event payload as delivered.
    """

    def __init__(self, client, daemonTaskVersion=None, deploymentConversationId=None, lifecycle=None, failureReason=None, isLimitBlocked=None, isTestRun=None, sequenceNumber=None, createdAt=None, updatedAt=None, source=None, eventType=None, payload=None):
        super().__init__(client, None)
        self.daemon_task_version = daemonTaskVersion
        self.deployment_conversation_id = deploymentConversationId
        self.lifecycle = lifecycle
        self.failure_reason = failureReason
        self.is_limit_blocked = isLimitBlocked
        self.is_test_run = isTestRun
        self.sequence_number = sequenceNumber
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.source = source
        self.event_type = eventType
        self.payload = payload
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'daemon_task_version': repr(self.daemon_task_version), f'deployment_conversation_id': repr(self.deployment_conversation_id), f'lifecycle': repr(self.lifecycle), f'failure_reason': repr(self.failure_reason), f'is_limit_blocked': repr(
            self.is_limit_blocked), f'is_test_run': repr(self.is_test_run), f'sequence_number': repr(self.sequence_number), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at), f'source': repr(self.source), f'event_type': repr(self.event_type), f'payload': repr(self.payload)}
        class_name = "DaemonTaskEvent"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'daemon_task_version': self.daemon_task_version, 'deployment_conversation_id': self.deployment_conversation_id, 'lifecycle': self.lifecycle, 'failure_reason': self.failure_reason, 'is_limit_blocked': self.is_limit_blocked,
                'is_test_run': self.is_test_run, 'sequence_number': self.sequence_number, 'created_at': self.created_at, 'updated_at': self.updated_at, 'source': self.source, 'event_type': self.event_type, 'payload': self.payload}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
