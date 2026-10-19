from .return_class import AbstractApiClass


class AudioListenerTranscriptSearchResult(AbstractApiClass):
    """
        A single audio listener transcript search hit. Hits are grouped per session, best session first.

        Args:
            client (ApiClient): An authenticated API Client instance
            deploymentConversationId (str): The unique identifier of the deployment conversation the transcript belongs to.
            name (str): The name of the deployment conversation.
            ownerUserId (str): The user id of the owner of the deployment conversation.
            createdAt (str): The timestamp at which the deployment conversation was created.
            lastEventCreatedAt (str): The timestamp at which the most recent event in the conversation was created.
            eventSequenceNumber (int): The sequence number of the matched transcript event within the conversation; absent for a name-only hit.
            eventType (str): The type of the matched transcript event.
            transcriptCreatedAt (str): The timestamp at which the matched transcript event was created.
            snippet (str): The matching transcript snippet.
            matchKind (str): What matched: 'line' (a transcript line), 'summary' (the session's latest summary) or 'name' (the session name only).
            nameMatched (bool): Whether the session name matches the query.
            sessionLineHits (int): How many transcript lines of the session match the query.
            speaker (str): Who said the matched line, when known.
    """

    def __init__(self, client, deploymentConversationId=None, name=None, ownerUserId=None, createdAt=None, lastEventCreatedAt=None, eventSequenceNumber=None, eventType=None, transcriptCreatedAt=None, snippet=None, matchKind=None, nameMatched=None, sessionLineHits=None, speaker=None):
        super().__init__(client, None)
        self.deployment_conversation_id = deploymentConversationId
        self.name = name
        self.owner_user_id = ownerUserId
        self.created_at = createdAt
        self.last_event_created_at = lastEventCreatedAt
        self.event_sequence_number = eventSequenceNumber
        self.event_type = eventType
        self.transcript_created_at = transcriptCreatedAt
        self.snippet = snippet
        self.match_kind = matchKind
        self.name_matched = nameMatched
        self.session_line_hits = sessionLineHits
        self.speaker = speaker
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'deployment_conversation_id': repr(self.deployment_conversation_id), f'name': repr(self.name), f'owner_user_id': repr(self.owner_user_id), f'created_at': repr(self.created_at), f'last_event_created_at': repr(self.last_event_created_at), f'event_sequence_number': repr(
            self.event_sequence_number), f'event_type': repr(self.event_type), f'transcript_created_at': repr(self.transcript_created_at), f'snippet': repr(self.snippet), f'match_kind': repr(self.match_kind), f'name_matched': repr(self.name_matched), f'session_line_hits': repr(self.session_line_hits), f'speaker': repr(self.speaker)}
        class_name = "AudioListenerTranscriptSearchResult"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'deployment_conversation_id': self.deployment_conversation_id, 'name': self.name, 'owner_user_id': self.owner_user_id, 'created_at': self.created_at, 'last_event_created_at': self.last_event_created_at, 'event_sequence_number': self.event_sequence_number,
                'event_type': self.event_type, 'transcript_created_at': self.transcript_created_at, 'snippet': self.snippet, 'match_kind': self.match_kind, 'name_matched': self.name_matched, 'session_line_hits': self.session_line_hits, 'speaker': self.speaker}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
