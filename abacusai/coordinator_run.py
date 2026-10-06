from .return_class import AbstractApiClass


class CoordinatorRun(AbstractApiClass):
    """
        A DeepAgent run a project's coordinator agent delegated (the project's Active Work rail)

        Args:
            client (ApiClient): An authenticated API Client instance
            taskId (str): The coordinator's id for the run.
            title (str): Short title of the brief.
            state (str): starting, running, finished, cancelled or gone.
            startedAt (str): ISO-8601 UTC time the run was started.
            deploymentConversationId (str): The project conversation the run happens in.
            computePointsUsed (int): Credits the run has consumed so far, when known.
            hasDeliverables (bool): Whether the run declared deliverable files.
    """

    def __init__(self, client, taskId=None, title=None, state=None, startedAt=None, deploymentConversationId=None, computePointsUsed=None, hasDeliverables=None):
        super().__init__(client, None)
        self.task_id = taskId
        self.title = title
        self.state = state
        self.started_at = startedAt
        self.deployment_conversation_id = deploymentConversationId
        self.compute_points_used = computePointsUsed
        self.has_deliverables = hasDeliverables
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'task_id': repr(self.task_id), f'title': repr(self.title), f'state': repr(self.state), f'started_at': repr(self.started_at), f'deployment_conversation_id': repr(
            self.deployment_conversation_id), f'compute_points_used': repr(self.compute_points_used), f'has_deliverables': repr(self.has_deliverables)}
        class_name = "CoordinatorRun"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'task_id': self.task_id, 'title': self.title, 'state': self.state, 'started_at': self.started_at, 'deployment_conversation_id':
                self.deployment_conversation_id, 'compute_points_used': self.compute_points_used, 'has_deliverables': self.has_deliverables}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
