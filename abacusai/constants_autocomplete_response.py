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
            diffThreshold (int): The diff operations threshold.
            derivativeThreshold (int): The derivative threshold for deletions
            defaultSurroundingLines (int): The default number of surrounding lines to include in the recently visited context.
            maxTrackedVisitChanges (int): The maximum number of recently visited ranges to track.
            selectionCooldownMs (int): The cooldown time in milliseconds for selection changes.
            viewingCooldownMs (int): The cooldown time in milliseconds for viewing changes.
            maxLines (int): The maximum number of lines to include in recently visited context.
            editCooldownMs (int): The cooldown time in milliseconds after last edit.
            scrollDebounceMs (int): The debounce time in milliseconds for scroll events.
            lspDeadline (int): The deadline in milliseconds for LSP context.
            diagnosticsThreshold (int): The max number of diagnostics to show.
            diagnosticEachThreshold (int): The max number of characters to show for each diagnostic type.
            numVsCodeSuggestions (int): The number of VS Code suggestions to show.
            minReindexingInterval (int): The minimum interval between reindexes in ms.
            minRefreshSummaryInterval (int): The minimum interval between refresh summary in ms.
            summaryBatchSize (int): The batch size for code summary in autocomplete.
            jobReorderInterval (int): The interval in ms to reorder jobs in the job queue for summary.
            stopRapidChanges (bool): Whether to stop rapid changes in autocomplete.
            delaySummaryBatches (int): The delay in ms between summary batches.
            maxSymbolsFuzzyMatch (int): The max number of symbols to fuzzy match.
            fuzzySymbolMatchThreshold (int): The threshold for fuzzy symbol match.
            symbolsCacheUpdateInterval (int): The interval in ms to update the symbols cache.
            symbolsStorageUpdateInterval (int): The interval in ms to update the symbols storage.
            editPredictionSimilarityThreshold (int): The threshold for edit prediction similarity.
            embeddingConstants (codellmembeddingconstants): Embedding constants
    """

    def __init__(self, client, maxPendingRequests=None, acceptanceDelay=None, debounceDelay=None, recordUserAction=None, validateSuggestion=None, validationLinesThreshold=None, maxTrackedRecentChanges=None, diffThreshold=None, derivativeThreshold=None, defaultSurroundingLines=None, maxTrackedVisitChanges=None, selectionCooldownMs=None, viewingCooldownMs=None, maxLines=None, editCooldownMs=None, scrollDebounceMs=None, lspDeadline=None, diagnosticsThreshold=None, diagnosticEachThreshold=None, numVsCodeSuggestions=None, minReindexingInterval=None, minRefreshSummaryInterval=None, summaryBatchSize=None, jobReorderInterval=None, stopRapidChanges=None, delaySummaryBatches=None, maxSymbolsFuzzyMatch=None, fuzzySymbolMatchThreshold=None, symbolsCacheUpdateInterval=None, symbolsStorageUpdateInterval=None, editPredictionSimilarityThreshold=None, embeddingConstants=None):
        super().__init__(client, None)
        self.max_pending_requests = maxPendingRequests
        self.acceptance_delay = acceptanceDelay
        self.debounce_delay = debounceDelay
        self.record_user_action = recordUserAction
        self.validate_suggestion = validateSuggestion
        self.validation_lines_threshold = validationLinesThreshold
        self.max_tracked_recent_changes = maxTrackedRecentChanges
        self.diff_threshold = diffThreshold
        self.derivative_threshold = derivativeThreshold
        self.default_surrounding_lines = defaultSurroundingLines
        self.max_tracked_visit_changes = maxTrackedVisitChanges
        self.selection_cooldown_ms = selectionCooldownMs
        self.viewing_cooldown_ms = viewingCooldownMs
        self.max_lines = maxLines
        self.edit_cooldown_ms = editCooldownMs
        self.scroll_debounce_ms = scrollDebounceMs
        self.lsp_deadline = lspDeadline
        self.diagnostics_threshold = diagnosticsThreshold
        self.diagnostic_each_threshold = diagnosticEachThreshold
        self.num_vs_code_suggestions = numVsCodeSuggestions
        self.min_reindexing_interval = minReindexingInterval
        self.min_refresh_summary_interval = minRefreshSummaryInterval
        self.summary_batch_size = summaryBatchSize
        self.job_reorder_interval = jobReorderInterval
        self.stop_rapid_changes = stopRapidChanges
        self.delay_summary_batches = delaySummaryBatches
        self.max_symbols_fuzzy_match = maxSymbolsFuzzyMatch
        self.fuzzy_symbol_match_threshold = fuzzySymbolMatchThreshold
        self.symbols_cache_update_interval = symbolsCacheUpdateInterval
        self.symbols_storage_update_interval = symbolsStorageUpdateInterval
        self.edit_prediction_similarity_threshold = editPredictionSimilarityThreshold
        self.embedding_constants = embeddingConstants
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'max_pending_requests': repr(self.max_pending_requests), f'acceptance_delay': repr(self.acceptance_delay), f'debounce_delay': repr(self.debounce_delay), f'record_user_action': repr(self.record_user_action), f'validate_suggestion': repr(self.validate_suggestion), f'validation_lines_threshold': repr(self.validation_lines_threshold), f'max_tracked_recent_changes': repr(self.max_tracked_recent_changes), f'diff_threshold': repr(self.diff_threshold), f'derivative_threshold': repr(self.derivative_threshold), f'default_surrounding_lines': repr(self.default_surrounding_lines), f'max_tracked_visit_changes': repr(self.max_tracked_visit_changes), f'selection_cooldown_ms': repr(self.selection_cooldown_ms), f'viewing_cooldown_ms': repr(self.viewing_cooldown_ms), f'max_lines': repr(self.max_lines), f'edit_cooldown_ms': repr(self.edit_cooldown_ms), f'scroll_debounce_ms': repr(self.scroll_debounce_ms), f'lsp_deadline': repr(self.lsp_deadline), f'diagnostics_threshold': repr(
            self.diagnostics_threshold), f'diagnostic_each_threshold': repr(self.diagnostic_each_threshold), f'num_vs_code_suggestions': repr(self.num_vs_code_suggestions), f'min_reindexing_interval': repr(self.min_reindexing_interval), f'min_refresh_summary_interval': repr(self.min_refresh_summary_interval), f'summary_batch_size': repr(self.summary_batch_size), f'job_reorder_interval': repr(self.job_reorder_interval), f'stop_rapid_changes': repr(self.stop_rapid_changes), f'delay_summary_batches': repr(self.delay_summary_batches), f'max_symbols_fuzzy_match': repr(self.max_symbols_fuzzy_match), f'fuzzy_symbol_match_threshold': repr(self.fuzzy_symbol_match_threshold), f'symbols_cache_update_interval': repr(self.symbols_cache_update_interval), f'symbols_storage_update_interval': repr(self.symbols_storage_update_interval), f'edit_prediction_similarity_threshold': repr(self.edit_prediction_similarity_threshold), f'embedding_constants': repr(self.embedding_constants)}
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
        resp = {'max_pending_requests': self.max_pending_requests, 'acceptance_delay': self.acceptance_delay, 'debounce_delay': self.debounce_delay, 'record_user_action': self.record_user_action, 'validate_suggestion': self.validate_suggestion, 'validation_lines_threshold': self.validation_lines_threshold, 'max_tracked_recent_changes': self.max_tracked_recent_changes, 'diff_threshold': self.diff_threshold, 'derivative_threshold': self.derivative_threshold, 'default_surrounding_lines': self.default_surrounding_lines, 'max_tracked_visit_changes': self.max_tracked_visit_changes, 'selection_cooldown_ms': self.selection_cooldown_ms, 'viewing_cooldown_ms': self.viewing_cooldown_ms, 'max_lines': self.max_lines, 'edit_cooldown_ms': self.edit_cooldown_ms, 'scroll_debounce_ms': self.scroll_debounce_ms, 'lsp_deadline': self.lsp_deadline, 'diagnostics_threshold': self.diagnostics_threshold,
                'diagnostic_each_threshold': self.diagnostic_each_threshold, 'num_vs_code_suggestions': self.num_vs_code_suggestions, 'min_reindexing_interval': self.min_reindexing_interval, 'min_refresh_summary_interval': self.min_refresh_summary_interval, 'summary_batch_size': self.summary_batch_size, 'job_reorder_interval': self.job_reorder_interval, 'stop_rapid_changes': self.stop_rapid_changes, 'delay_summary_batches': self.delay_summary_batches, 'max_symbols_fuzzy_match': self.max_symbols_fuzzy_match, 'fuzzy_symbol_match_threshold': self.fuzzy_symbol_match_threshold, 'symbols_cache_update_interval': self.symbols_cache_update_interval, 'symbols_storage_update_interval': self.symbols_storage_update_interval, 'edit_prediction_similarity_threshold': self.edit_prediction_similarity_threshold, 'embedding_constants': self.embedding_constants}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
