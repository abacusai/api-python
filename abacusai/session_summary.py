from .return_class import AbstractApiClass


class SessionSummary(AbstractApiClass):
    """
        A session summary

        Args:
            client (ApiClient): An authenticated API Client instance
            summary (str): The summary of the session.
            templateName (str): The name of the Listener template the session ran with, if any.
            meetingType (str): The meeting type of that template.
            meetingNotes (str): The session's notes for this meeting, returned only to the session's owner.
            meeting (dict): The calendar event the session is attached to (title, times, provider, recurring), returned only to the owner.
            previousMeeting (dict): The owner's previous session of the same meeting (id, name, created_at_ms), returned only to the owner.
            feedback (dict): The owner's rating of the session (rating 1 to 5, comment, at_ms), returned only to the owner.
            summaryShared (bool): Whether this viewer may read the summary (always for the owner and org admins); the summary is empty otherwise.
            transcriptShared (bool): Whether this viewer may read the transcript.
            chatShared (bool): Whether this viewer may read the session's chat.
            shareParts (dict): Which parts the owner shares, {summary, transcript, chat}; returned only to the owner.
            canEdit (bool): Whether the caller may save changes to the session now; returned only to the owner, and only where editing is on.
            editBlocked (str): Why not, when can_edit is false: recording, paused or finishing.
            edits (dict): The owner's saved changes so far: version (what a save must send), at, lines, summary_seqs; returned only to the owner.
            summarySeq (int): The sequence number of the summary row shown, which a saved summary must name; returned only to the owner.
            editedAt (str): When the owner last saved a change, if ever.
            summaryEdited (bool): Whether the summary shown was written by the owner rather than the model.
            summaryKind (str): Where the summary shown came from: model, edit (the owner's hand), rewrite (the owner's instruction).
            summaryAt (str): When that version was written.
            summaryHistory (list[dict]): The session's summary versions, newest first ({seq, at, kind, base_seq, instruction, preview, from_seq}); returned only to the owner.
            canRewriteSummary (bool): Whether the owner may rewrite the summary with an instruction now; returned only to the owner.
    """

    def __init__(self, client, summary=None, templateName=None, meetingType=None, meetingNotes=None, meeting=None, previousMeeting=None, feedback=None, summaryShared=None, transcriptShared=None, chatShared=None, shareParts=None, canEdit=None, editBlocked=None, edits=None, summarySeq=None, editedAt=None, summaryEdited=None, summaryKind=None, summaryAt=None, summaryHistory=None, canRewriteSummary=None):
        super().__init__(client, None)
        self.summary = summary
        self.template_name = templateName
        self.meeting_type = meetingType
        self.meeting_notes = meetingNotes
        self.meeting = meeting
        self.previous_meeting = previousMeeting
        self.feedback = feedback
        self.summary_shared = summaryShared
        self.transcript_shared = transcriptShared
        self.chat_shared = chatShared
        self.share_parts = shareParts
        self.can_edit = canEdit
        self.edit_blocked = editBlocked
        self.edits = edits
        self.summary_seq = summarySeq
        self.edited_at = editedAt
        self.summary_edited = summaryEdited
        self.summary_kind = summaryKind
        self.summary_at = summaryAt
        self.summary_history = summaryHistory
        self.can_rewrite_summary = canRewriteSummary
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'summary': repr(self.summary), f'template_name': repr(self.template_name), f'meeting_type': repr(self.meeting_type), f'meeting_notes': repr(self.meeting_notes), f'meeting': repr(self.meeting), f'previous_meeting': repr(self.previous_meeting), f'feedback': repr(self.feedback), f'summary_shared': repr(self.summary_shared), f'transcript_shared': repr(self.transcript_shared), f'chat_shared': repr(self.chat_shared), f'share_parts': repr(
            self.share_parts), f'can_edit': repr(self.can_edit), f'edit_blocked': repr(self.edit_blocked), f'edits': repr(self.edits), f'summary_seq': repr(self.summary_seq), f'edited_at': repr(self.edited_at), f'summary_edited': repr(self.summary_edited), f'summary_kind': repr(self.summary_kind), f'summary_at': repr(self.summary_at), f'summary_history': repr(self.summary_history), f'can_rewrite_summary': repr(self.can_rewrite_summary)}
        class_name = "SessionSummary"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'summary': self.summary, 'template_name': self.template_name, 'meeting_type': self.meeting_type, 'meeting_notes': self.meeting_notes, 'meeting': self.meeting, 'previous_meeting': self.previous_meeting, 'feedback': self.feedback, 'summary_shared': self.summary_shared, 'transcript_shared': self.transcript_shared, 'chat_shared': self.chat_shared,
                'share_parts': self.share_parts, 'can_edit': self.can_edit, 'edit_blocked': self.edit_blocked, 'edits': self.edits, 'summary_seq': self.summary_seq, 'edited_at': self.edited_at, 'summary_edited': self.summary_edited, 'summary_kind': self.summary_kind, 'summary_at': self.summary_at, 'summary_history': self.summary_history, 'can_rewrite_summary': self.can_rewrite_summary}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
