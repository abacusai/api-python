from .return_class import AbstractApiClass


class AudioListenerSummaryRewrite(AbstractApiClass):
    """
        The outcome of rewriting a Listener session's summary from the owner's instruction.

        Args:
            client (ApiClient): An authenticated API Client instance
            summary (str): The summary the session now shows: the rewritten one, or the current one on a conflict.
            summarySeq (int): Its summary row, which the next rewrite or save must name.
            summaryKind (str): Where it came from: model, edit or rewrite.
            summaryAt (str): When that version was written.
            version (int): The session's edits version after the rewrite.
            editedAt (str): When the rewrite landed.
            summaryHistory (list[dict]): The session's summary versions, newest first.
            conflict (bool): True when the summary changed elsewhere since the client read it: nothing was written, summary is the current one.
            unchanged (bool): True when the instruction left the summary as it was: nothing was written.
            running (bool): True when a rewrite of this session is already running (another device, or a retry): nothing was started.
    """

    def __init__(self, client, summary=None, summarySeq=None, summaryKind=None, summaryAt=None, version=None, editedAt=None, summaryHistory=None, conflict=None, unchanged=None, running=None):
        super().__init__(client, None)
        self.summary = summary
        self.summary_seq = summarySeq
        self.summary_kind = summaryKind
        self.summary_at = summaryAt
        self.version = version
        self.edited_at = editedAt
        self.summary_history = summaryHistory
        self.conflict = conflict
        self.unchanged = unchanged
        self.running = running
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'summary': repr(self.summary), f'summary_seq': repr(self.summary_seq), f'summary_kind': repr(self.summary_kind), f'summary_at': repr(self.summary_at), f'version': repr(
            self.version), f'edited_at': repr(self.edited_at), f'summary_history': repr(self.summary_history), f'conflict': repr(self.conflict), f'unchanged': repr(self.unchanged), f'running': repr(self.running)}
        class_name = "AudioListenerSummaryRewrite"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'summary': self.summary, 'summary_seq': self.summary_seq, 'summary_kind': self.summary_kind, 'summary_at': self.summary_at, 'version': self.version,
                'edited_at': self.edited_at, 'summary_history': self.summary_history, 'conflict': self.conflict, 'unchanged': self.unchanged, 'running': self.running}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
