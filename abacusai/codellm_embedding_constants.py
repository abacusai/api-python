from .return_class import AbstractApiClass


class CodellmEmbeddingConstants(AbstractApiClass):
    """
        A dictionary of constants to be used in the autocomplete.

        Args:
            client (ApiClient): An authenticated API Client instance
            maxSupportedWorkspaceFiles (int): Max supported workspace files
            maxSupportedWorkspaceChunks (int): Max supported workspace chunks
            maxConcurrentRequests (int): Max concurrent requests
            fileExtensionToChunkingScheme (dict): Map between the file extensions and their chunking schema
            idleTimeoutSeconds (int): The idle timeout without any activity before the workspace is refreshed.
    """

    def __init__(self, client, maxSupportedWorkspaceFiles=None, maxSupportedWorkspaceChunks=None, maxConcurrentRequests=None, fileExtensionToChunkingScheme=None, idleTimeoutSeconds=None):
        super().__init__(client, None)
        self.max_supported_workspace_files = maxSupportedWorkspaceFiles
        self.max_supported_workspace_chunks = maxSupportedWorkspaceChunks
        self.max_concurrent_requests = maxConcurrentRequests
        self.file_extension_to_chunking_scheme = fileExtensionToChunkingScheme
        self.idle_timeout_seconds = idleTimeoutSeconds
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'max_supported_workspace_files': repr(self.max_supported_workspace_files), f'max_supported_workspace_chunks': repr(self.max_supported_workspace_chunks), f'max_concurrent_requests': repr(
            self.max_concurrent_requests), f'file_extension_to_chunking_scheme': repr(self.file_extension_to_chunking_scheme), f'idle_timeout_seconds': repr(self.idle_timeout_seconds)}
        class_name = "CodellmEmbeddingConstants"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'max_supported_workspace_files': self.max_supported_workspace_files, 'max_supported_workspace_chunks': self.max_supported_workspace_chunks,
                'max_concurrent_requests': self.max_concurrent_requests, 'file_extension_to_chunking_scheme': self.file_extension_to_chunking_scheme, 'idle_timeout_seconds': self.idle_timeout_seconds}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
