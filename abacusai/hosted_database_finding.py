from .return_class import AbstractApiClass


class HostedDatabaseFinding(AbstractApiClass):
    """
        A database-health finding for a web app's database (_getHostedDatabaseHealth), as the hourly scan last recorded it.

        Args:
            client (ApiClient): An authenticated API Client instance
            findingId (str): kind:table:column, stable while the finding stands
            kind (str): missing_index | repeated_query | substring_search
            severity (str): LOW | MEDIUM | HIGH | CRITICAL
            query (str): The statement, normalized, when the finding comes from statement statistics
            table (str): The table involved
            column (str): The column involved, when there is one
            execSecondsPerHour (int): Execution time this statement costs per hour
            callsPerHour (int): How often it runs
            rowsPerCall (float): Rows it returns per call
            blocksPerCall (float): Buffers it reads per call
            meanMs (float): Mean execution time per call
            rows (int): Rows in the table, for a missing index
            firstSeen (str): When the scan first recorded this finding
            lastSeen (str): When the scan last saw it
            fixPrompt (str): What the agent is asked to change when the finding is sent to it
    """

    def __init__(self, client, findingId=None, kind=None, severity=None, query=None, table=None, column=None, execSecondsPerHour=None, callsPerHour=None, rowsPerCall=None, blocksPerCall=None, meanMs=None, rows=None, firstSeen=None, lastSeen=None, fixPrompt=None):
        super().__init__(client, None)
        self.finding_id = findingId
        self.kind = kind
        self.severity = severity
        self.query = query
        self.table = table
        self.column = column
        self.exec_seconds_per_hour = execSecondsPerHour
        self.calls_per_hour = callsPerHour
        self.rows_per_call = rowsPerCall
        self.blocks_per_call = blocksPerCall
        self.mean_ms = meanMs
        self.rows = rows
        self.first_seen = firstSeen
        self.last_seen = lastSeen
        self.fix_prompt = fixPrompt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'finding_id': repr(self.finding_id), f'kind': repr(self.kind), f'severity': repr(self.severity), f'query': repr(self.query), f'table': repr(self.table), f'column': repr(self.column), f'exec_seconds_per_hour': repr(self.exec_seconds_per_hour), f'calls_per_hour': repr(
            self.calls_per_hour), f'rows_per_call': repr(self.rows_per_call), f'blocks_per_call': repr(self.blocks_per_call), f'mean_ms': repr(self.mean_ms), f'rows': repr(self.rows), f'first_seen': repr(self.first_seen), f'last_seen': repr(self.last_seen), f'fix_prompt': repr(self.fix_prompt)}
        class_name = "HostedDatabaseFinding"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'finding_id': self.finding_id, 'kind': self.kind, 'severity': self.severity, 'query': self.query, 'table': self.table, 'column': self.column, 'exec_seconds_per_hour': self.exec_seconds_per_hour, 'calls_per_hour': self.calls_per_hour,
                'rows_per_call': self.rows_per_call, 'blocks_per_call': self.blocks_per_call, 'mean_ms': self.mean_ms, 'rows': self.rows, 'first_seen': self.first_seen, 'last_seen': self.last_seen, 'fix_prompt': self.fix_prompt}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
