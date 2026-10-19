from .return_class import AbstractApiClass


class SharedAudioListenerSession(AbstractApiClass):
    """
        A Listener session as its web page shows it, to anyone holding the session's link or to a viewer it is shared with.

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The session's title.
            ownerName (str): The name of the person who recorded it, as snapshotted when the link was made; only on the public link path.
            createdAt (str): When the session was created.
            lastLineAt (str): When the last line of speech was recorded.
            isLive (bool): Whether the session is still recording.
            summary (str): The session's summary, as stored markdown.
            transcripts (list): The lines of speech: role (user or others), content, created_at and speaker when one was named.
            speakers (list): The far-end speaker names found in the session.
            language (str): The language the meeting was held in, when known.
            chat (list): The owner's chat with the assistant about the session, {role: user|assistant, content}, when shared; else empty.
            summaryShared (bool): False when the owner shares the session without its summary, which is then empty.
            transcriptShared (bool): False when the owner shares the session without its transcript: transcripts and speakers are then empty.
            chatShared (bool): Whether the chat is shared with this viewer.
            shareParts (dict): Which parts viewers get, {summary, transcript, chat}; only for the owner or an admin, once someone has access.
            editedAt (str): When the owner last saved a change to the session, if ever; each changed line carries its own edited_at.
    """

    def __init__(self, client, name=None, ownerName=None, createdAt=None, lastLineAt=None, isLive=None, summary=None, transcripts=None, speakers=None, language=None, chat=None, summaryShared=None, transcriptShared=None, chatShared=None, shareParts=None, editedAt=None):
        super().__init__(client, None)
        self.name = name
        self.owner_name = ownerName
        self.created_at = createdAt
        self.last_line_at = lastLineAt
        self.is_live = isLive
        self.summary = summary
        self.transcripts = transcripts
        self.speakers = speakers
        self.language = language
        self.chat = chat
        self.summary_shared = summaryShared
        self.transcript_shared = transcriptShared
        self.chat_shared = chatShared
        self.share_parts = shareParts
        self.edited_at = editedAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'owner_name': repr(self.owner_name), f'created_at': repr(self.created_at), f'last_line_at': repr(self.last_line_at), f'is_live': repr(self.is_live), f'summary': repr(self.summary), f'transcripts': repr(self.transcripts), f'speakers': repr(
            self.speakers), f'language': repr(self.language), f'chat': repr(self.chat), f'summary_shared': repr(self.summary_shared), f'transcript_shared': repr(self.transcript_shared), f'chat_shared': repr(self.chat_shared), f'share_parts': repr(self.share_parts), f'edited_at': repr(self.edited_at)}
        class_name = "SharedAudioListenerSession"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'owner_name': self.owner_name, 'created_at': self.created_at, 'last_line_at': self.last_line_at, 'is_live': self.is_live, 'summary': self.summary, 'transcripts': self.transcripts, 'speakers': self.speakers,
                'language': self.language, 'chat': self.chat, 'summary_shared': self.summary_shared, 'transcript_shared': self.transcript_shared, 'chat_shared': self.chat_shared, 'share_parts': self.share_parts, 'edited_at': self.edited_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
