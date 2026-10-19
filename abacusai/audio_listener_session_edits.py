from .return_class import AbstractApiClass


class AudioListenerSessionEdits(AbstractApiClass):
    """
        The outcome of saving the owner's changes to a Listener session.

        Args:
            client (ApiClient): An authenticated API Client instance
            conflict (bool): True when the session was changed elsewhere since the client opened it: nothing was saved, version is the current one.
            version (int): The session's edits version after the save (or the current one on a conflict).
            editedAt (str): When the save landed.
            linesChanged (int): Lines whose text or speaker changed.
            speakersChanged (int): Lines whose speaker changed.
            summarySaved (bool): Whether the summary was stored.
            summarySkipped (bool): True when the summary was not stored because a newer summary row existed.
            summarySeq (int): The summary row the session now shows, which the next saved summary must name; only when a summary was sent.
            summary (str): The summary the session now shows, when it is not the one the client sent (a skipped save, a restore).
            summaryKind (str): Where that summary came from: model, edit or rewrite.
            summaryAt (str): When that version was written.
            summaryHistory (list[dict]): The session's summary versions after the save, newest first.
    """

    def __init__(self, client, conflict=None, version=None, editedAt=None, linesChanged=None, speakersChanged=None, summarySaved=None, summarySkipped=None, summarySeq=None, summary=None, summaryKind=None, summaryAt=None, summaryHistory=None):
        super().__init__(client, None)
        self.conflict = conflict
        self.version = version
        self.edited_at = editedAt
        self.lines_changed = linesChanged
        self.speakers_changed = speakersChanged
        self.summary_saved = summarySaved
        self.summary_skipped = summarySkipped
        self.summary_seq = summarySeq
        self.summary = summary
        self.summary_kind = summaryKind
        self.summary_at = summaryAt
        self.summary_history = summaryHistory
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'conflict': repr(self.conflict), f'version': repr(self.version), f'edited_at': repr(self.edited_at), f'lines_changed': repr(self.lines_changed), f'speakers_changed': repr(self.speakers_changed), f'summary_saved': repr(
            self.summary_saved), f'summary_skipped': repr(self.summary_skipped), f'summary_seq': repr(self.summary_seq), f'summary': repr(self.summary), f'summary_kind': repr(self.summary_kind), f'summary_at': repr(self.summary_at), f'summary_history': repr(self.summary_history)}
        class_name = "AudioListenerSessionEdits"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'conflict': self.conflict, 'version': self.version, 'edited_at': self.edited_at, 'lines_changed': self.lines_changed, 'speakers_changed': self.speakers_changed, 'summary_saved': self.summary_saved,
                'summary_skipped': self.summary_skipped, 'summary_seq': self.summary_seq, 'summary': self.summary, 'summary_kind': self.summary_kind, 'summary_at': self.summary_at, 'summary_history': self.summary_history}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
