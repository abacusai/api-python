from .return_class import AbstractApiClass


class DaemonTaskInstance(AbstractApiClass):
    """
        A daemon task instance representing a single run of a daemon task.

        Args:
            client (ApiClient): An authenticated API Client instance
            daemonTaskVersion (str): The unique identifier for this task instance.
            daemonTaskId (str): The id of the parent daemon task.
            deploymentConversationId (str): The deployment conversation id for this instance.
            lifecycle (str): The lifecycle status (PENDING, EXECUTING, COMPLETED, FAILED).
            createdAt (str): When this instance was created.
            lifecycleInfo (dict): Additional lifecycle information (e.g., error details).
    """

    def __init__(self, client, daemonTaskVersion=None, daemonTaskId=None, deploymentConversationId=None, lifecycle=None, createdAt=None, lifecycleInfo=None):
        super().__init__(client, None)
        self.daemon_task_version = daemonTaskVersion
        self.daemon_task_id = daemonTaskId
        self.deployment_conversation_id = deploymentConversationId
        self.lifecycle = lifecycle
        self.created_at = createdAt
        self.lifecycle_info = lifecycleInfo
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'daemon_task_version': repr(self.daemon_task_version), f'daemon_task_id': repr(self.daemon_task_id), f'deployment_conversation_id': repr(
            self.deployment_conversation_id), f'lifecycle': repr(self.lifecycle), f'created_at': repr(self.created_at), f'lifecycle_info': repr(self.lifecycle_info)}
        class_name = "DaemonTaskInstance"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'daemon_task_version': self.daemon_task_version, 'daemon_task_id': self.daemon_task_id, 'deployment_conversation_id':
                self.deployment_conversation_id, 'lifecycle': self.lifecycle, 'created_at': self.created_at, 'lifecycle_info': self.lifecycle_info}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
