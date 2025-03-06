from .return_class import AbstractApiClass


class ConstantsAutocompleteResponse(AbstractApiClass):
    """
        A dictionary of constants to be used in the autocomplete.

        Args:
            client (ApiClient): An authenticated API Client instance
            maxPendingRequests (int): The maximum number of pending requests.
            acceptanceDelay (int): The acceptance delay.
            debounceDelay (int): The debounce delay.
            recordUserAction (bool): Whether to record user action.
            validateSuggestion (bool): Whether to validate the suggestion.
            validationLinesThreshold (int): The number of lines to validate the suggestion.
            maxTrackedRecentChanges (int): The maximum number of recent file changes to track.
    """

    def __init__(self, client, maxPendingRequests=None, acceptanceDelay=None, debounceDelay=None, recordUserAction=None, validateSuggestion=None, validationLinesThreshold=None, maxTrackedRecentChanges=None):
        super().__init__(client, None)
        self.max_pending_requests = maxPendingRequests
        self.acceptance_delay = acceptanceDelay
        self.debounce_delay = debounceDelay
        self.record_user_action = recordUserAction
        self.validate_suggestion = validateSuggestion
        self.validation_lines_threshold = validationLinesThreshold
        self.max_tracked_recent_changes = maxTrackedRecentChanges
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'max_pending_requests': repr(self.max_pending_requests), f'acceptance_delay': repr(self.acceptance_delay), f'debounce_delay': repr(self.debounce_delay), f'record_user_action': repr(
            self.record_user_action), f'validate_suggestion': repr(self.validate_suggestion), f'validation_lines_threshold': repr(self.validation_lines_threshold), f'max_tracked_recent_changes': repr(self.max_tracked_recent_changes)}
        class_name = "ConstantsAutocompleteResponse"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'max_pending_requests': self.max_pending_requests, 'acceptance_delay': self.acceptance_delay, 'debounce_delay': self.debounce_delay, 'record_user_action': self.record_user_action,
                'validate_suggestion': self.validate_suggestion, 'validation_lines_threshold': self.validation_lines_threshold, 'max_tracked_recent_changes': self.max_tracked_recent_changes}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
