from .return_class import AbstractApiClass


class McpServerConnection(AbstractApiClass):
    """
        Model Context Protocol Server Connection, as the MCP Servers page lists and edits it: the server, its sign-in kind, which

        Args:
            client (ApiClient): An authenticated API Client instance
            mcpServerConnectionId (id): the id of the MCP server connection.
            name (str): The name of the MCP server.
            config (dict): a dictionary containing the config for the MCP server.
            authType (str): the auth type for the MCP server.
            inactive (bool): whether the MCP server is inactive.
            tools (list): the tools for the MCP server, each with its name, title, description and annotations.
            errorMsg (str): the error message for the MCP server.
            metadata (dict): tool_permissions, the tools the user turned off.
    """

    def __init__(self, client, mcpServerConnectionId=None, name=None, config=None, authType=None, inactive=None, tools=None, errorMsg=None, metadata=None):
        super().__init__(client, mcpServerConnectionId)
        self.mcp_server_connection_id = mcpServerConnectionId
        self.name = name
        self.config = config
        self.auth_type = authType
        self.inactive = inactive
        self.tools = tools
        self.error_msg = errorMsg
        self.metadata = metadata
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'mcp_server_connection_id': repr(self.mcp_server_connection_id), f'name': repr(self.name), f'config': repr(self.config), f'auth_type': repr(
            self.auth_type), f'inactive': repr(self.inactive), f'tools': repr(self.tools), f'error_msg': repr(self.error_msg), f'metadata': repr(self.metadata)}
        class_name = "McpServerConnection"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'mcp_server_connection_id': self.mcp_server_connection_id, 'name': self.name, 'config': self.config,
                'auth_type': self.auth_type, 'inactive': self.inactive, 'tools': self.tools, 'error_msg': self.error_msg, 'metadata': self.metadata}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
