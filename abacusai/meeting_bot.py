from .return_class import AbstractApiClass


class MeetingBot(AbstractApiClass):
    """
        A meeting the user sent the notetaker to (cluster_proxy/meeting_bot.py).

        Args:
            client (ApiClient): An authenticated API Client instance
            daemonTaskId (str): The id of the task that runs the notetaker.
            state (str): scheduled, joining, waiting_for_admission, in_meeting, finished, not_admitted, denied, failed or cancelled.
            title (str): The meeting's title, when known.
            meetingUrl (str): The meeting link.
            platform (str): The meeting app (meet).
            startAtMs (int): When the meeting starts, epoch milliseconds.
            endAtMs (int): When the meeting ends, epoch milliseconds, when known.
            createdAtMs (int): When the notetaker was booked, epoch milliseconds.
            sessionDeploymentConversationId (str): The Listener session the recap is in.
            sessionName (str): The session's name, as the Listener named it.
            webChatPath (str): The session's web chat, once it was opened.
            leftReason (str): Why the notetaker left (meeting_ended, removed, alone, scheduled_end, cancelled, max_duration).
            error (str): What went wrong, for a failed one.
            account (str): How the notetaker was in the call: bot (its Google account) or guest, once it ran.
            botEmail (str): The notetaker's Google account, the address to invite to a meeting; absent when it joins as a guest.
            invited (bool): On booking from a calendar event: whether the notetaker's account was put on the event's guest list.
    """

    def __init__(self, client, daemonTaskId=None, state=None, title=None, meetingUrl=None, platform=None, startAtMs=None, endAtMs=None, createdAtMs=None, sessionDeploymentConversationId=None, sessionName=None, webChatPath=None, leftReason=None, error=None, account=None, botEmail=None, invited=None):
        super().__init__(client, None)
        self.daemon_task_id = daemonTaskId
        self.state = state
        self.title = title
        self.meeting_url = meetingUrl
        self.platform = platform
        self.start_at_ms = startAtMs
        self.end_at_ms = endAtMs
        self.created_at_ms = createdAtMs
        self.session_deployment_conversation_id = sessionDeploymentConversationId
        self.session_name = sessionName
        self.web_chat_path = webChatPath
        self.left_reason = leftReason
        self.error = error
        self.account = account
        self.bot_email = botEmail
        self.invited = invited
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'daemon_task_id': repr(self.daemon_task_id), f'state': repr(self.state), f'title': repr(self.title), f'meeting_url': repr(self.meeting_url), f'platform': repr(self.platform), f'start_at_ms': repr(self.start_at_ms), f'end_at_ms': repr(self.end_at_ms), f'created_at_ms': repr(self.created_at_ms), f'session_deployment_conversation_id': repr(
            self.session_deployment_conversation_id), f'session_name': repr(self.session_name), f'web_chat_path': repr(self.web_chat_path), f'left_reason': repr(self.left_reason), f'error': repr(self.error), f'account': repr(self.account), f'bot_email': repr(self.bot_email), f'invited': repr(self.invited)}
        class_name = "MeetingBot"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'daemon_task_id': self.daemon_task_id, 'state': self.state, 'title': self.title, 'meeting_url': self.meeting_url, 'platform': self.platform, 'start_at_ms': self.start_at_ms, 'end_at_ms': self.end_at_ms, 'created_at_ms': self.created_at_ms,
                'session_deployment_conversation_id': self.session_deployment_conversation_id, 'session_name': self.session_name, 'web_chat_path': self.web_chat_path, 'left_reason': self.left_reason, 'error': self.error, 'account': self.account, 'bot_email': self.bot_email, 'invited': self.invited}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
