from .return_class import AbstractApiClass


class CoordinatorRun(AbstractApiClass):
    """
        A DeepAgent run an agent delegated (a project's Active Work rail, the run cards in the agent's chat)

        Args:
            client (ApiClient): An authenticated API Client instance
            taskId (str): The coordinator's id for the run.
            title (str): Short title of the brief.
            state (str): starting, running, finished, cancelled or gone.
            startedAt (str): ISO-8601 UTC time the run was started.
            deploymentConversationId (str): The project conversation the run happens in.
            computePointsUsed (int): Credits the run has consumed so far, when known.
            hasDeliverables (bool): Whether the run declared deliverable files.
            turns (int): Messages the run has taken from its agent, the brief included.
            maxTurns (int): Messages the run can take from its agent.
            waiting (bool): Whether the run stopped on a question for its agent (state reads running).
            sentBack (bool): Whether the run is working on a message its agent sent on its own (e.g. a result it rejected).
            summary (str): The latest finished turn's result as shown to the user (the run's final message, condensed when long); null while the run is mid-turn.
            deliverables (list): The files that turn declared, {path, url} each; empty while the run is mid-turn.
            plan (list): While the run works, its own plan for the current turn ({id, task, status} per step, status pending, in_progress or completed); null otherwise or before it writes one.
            step (str): While the run works, the step it is on; null otherwise.
            turnStartedAt (str): ISO-8601 UTC time the run's current or latest turn started.
    """

    def __init__(self, client, taskId=None, title=None, state=None, startedAt=None, deploymentConversationId=None, computePointsUsed=None, hasDeliverables=None, turns=None, maxTurns=None, waiting=None, sentBack=None, summary=None, deliverables=None, plan=None, step=None, turnStartedAt=None):
        super().__init__(client, None)
        self.task_id = taskId
        self.title = title
        self.state = state
        self.started_at = startedAt
        self.deployment_conversation_id = deploymentConversationId
        self.compute_points_used = computePointsUsed
        self.has_deliverables = hasDeliverables
        self.turns = turns
        self.max_turns = maxTurns
        self.waiting = waiting
        self.sent_back = sentBack
        self.summary = summary
        self.deliverables = deliverables
        self.plan = plan
        self.step = step
        self.turn_started_at = turnStartedAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'task_id': repr(self.task_id), f'title': repr(self.title), f'state': repr(self.state), f'started_at': repr(self.started_at), f'deployment_conversation_id': repr(self.deployment_conversation_id), f'compute_points_used': repr(self.compute_points_used), f'has_deliverables': repr(self.has_deliverables), f'turns': repr(
            self.turns), f'max_turns': repr(self.max_turns), f'waiting': repr(self.waiting), f'sent_back': repr(self.sent_back), f'summary': repr(self.summary), f'deliverables': repr(self.deliverables), f'plan': repr(self.plan), f'step': repr(self.step), f'turn_started_at': repr(self.turn_started_at)}
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
        resp = {'task_id': self.task_id, 'title': self.title, 'state': self.state, 'started_at': self.started_at, 'deployment_conversation_id': self.deployment_conversation_id, 'compute_points_used': self.compute_points_used, 'has_deliverables': self.has_deliverables,
                'turns': self.turns, 'max_turns': self.max_turns, 'waiting': self.waiting, 'sent_back': self.sent_back, 'summary': self.summary, 'deliverables': self.deliverables, 'plan': self.plan, 'step': self.step, 'turn_started_at': self.turn_started_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
