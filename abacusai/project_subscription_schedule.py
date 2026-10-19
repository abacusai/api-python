from .return_class import AbstractApiClass


class ProjectSubscriptionSchedule(AbstractApiClass):
    """
        One schedule that wakes a project's coordinator (its check-in or a named task)

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): Display name.
            kind (str): heartbeat or task.
            cadence (str): Human-readable cadence, e.g. "daily at 09:00" or "on event".
            cron (str): The 5-field cron expression, for a free-form cron schedule only.
            timezone (str): IANA zone the cadence is expressed in.
            maxRuns (int): How many times the schedule fires in total, when it is capped.
            state (str): The schedule's own lifecycle: ACTIVE, COMPLETED (it will not fire again) or PAUSED.
            daemonTaskId (str): The backing scheduled task.
            lastRunLifecycle (str): Lifecycle of the newest run, when any.
            lastRunAt (str): ISO-8601 time of the newest run, when any.
            webhookUrl (str): Signed URL that fires an event task.
    """

    def __init__(self, client, name=None, kind=None, cadence=None, cron=None, timezone=None, maxRuns=None, state=None, daemonTaskId=None, lastRunLifecycle=None, lastRunAt=None, webhookUrl=None):
        super().__init__(client, None)
        self.name = name
        self.kind = kind
        self.cadence = cadence
        self.cron = cron
        self.timezone = timezone
        self.max_runs = maxRuns
        self.state = state
        self.daemon_task_id = daemonTaskId
        self.last_run_lifecycle = lastRunLifecycle
        self.last_run_at = lastRunAt
        self.webhook_url = webhookUrl
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'kind': repr(self.kind), f'cadence': repr(self.cadence), f'cron': repr(self.cron), f'timezone': repr(self.timezone), f'max_runs': repr(self.max_runs), f'state': repr(
            self.state), f'daemon_task_id': repr(self.daemon_task_id), f'last_run_lifecycle': repr(self.last_run_lifecycle), f'last_run_at': repr(self.last_run_at), f'webhook_url': repr(self.webhook_url)}
        class_name = "ProjectSubscriptionSchedule"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'kind': self.kind, 'cadence': self.cadence, 'cron': self.cron, 'timezone': self.timezone, 'max_runs': self.max_runs, 'state': self.state,
                'daemon_task_id': self.daemon_task_id, 'last_run_lifecycle': self.last_run_lifecycle, 'last_run_at': self.last_run_at, 'webhook_url': self.webhook_url}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
