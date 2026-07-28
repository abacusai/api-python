from .return_class import AbstractApiClass


class WebAppIssue(AbstractApiClass):
    """
        A self-heal issue detected in a deployed web app's production logs.

        Args:
            client (ApiClient): An authenticated API Client instance
            webAppIssueId (id): The issue ID
            summary (str): Plain-English one-line summary of the error
            severity (str): LOW | MEDIUM | HIGH | CRITICAL
            source (str): SERVER | CLIENT — where the error originated
            sampleTrace (str): A sample error trace for this issue
            occurrenceCount (int): Number of times this error has been seen
            status (str): OPEN | FIXING | AWAITING_USER | RESOLVED | WONT_DO
            lastSeen (str): When the error was last seen
            createdAt (str): When the issue was first detected
    """

    def __init__(self, client, webAppIssueId=None, summary=None, severity=None, source=None, sampleTrace=None, occurrenceCount=None, status=None, lastSeen=None, createdAt=None):
        super().__init__(client, webAppIssueId)
        self.web_app_issue_id = webAppIssueId
        self.summary = summary
        self.severity = severity
        self.source = source
        self.sample_trace = sampleTrace
        self.occurrence_count = occurrenceCount
        self.status = status
        self.last_seen = lastSeen
        self.created_at = createdAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'web_app_issue_id': repr(self.web_app_issue_id), f'summary': repr(self.summary), f'severity': repr(self.severity), f'source': repr(self.source), f'sample_trace': repr(
            self.sample_trace), f'occurrence_count': repr(self.occurrence_count), f'status': repr(self.status), f'last_seen': repr(self.last_seen), f'created_at': repr(self.created_at)}
        class_name = "WebAppIssue"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'web_app_issue_id': self.web_app_issue_id, 'summary': self.summary, 'severity': self.severity, 'source': self.source, 'sample_trace': self.sample_trace,
                'occurrence_count': self.occurrence_count, 'status': self.status, 'last_seen': self.last_seen, 'created_at': self.created_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
