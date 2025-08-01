from .return_class import AbstractApiClass


class McpServerConnection(AbstractApiClass):
    """
        Model Context Protocol Server Connection

        Args:
            client (ApiClient): An authenticated API Client instance
            mcpServerConnectionId (id): the id of the MCP server connection.
            createdAt (str): the date and time the MCP server connection was created.
            updatedAt (str): the date and time the MCP server connection was updated.
            name (str): The name of the MCP server.
            config (dict): a dictionary containing the config for the MCP server.
            description (str): description of what the MCP server does.
            transport (str): the transport type for the MCP server.
            authType (str): the auth type for the MCP server.
            externalConnectionId (id): the external connection id for the MCP server.
            inactive (bool): whether the MCP server is inactive.
            tools (list): the tools for the MCP server.
            errorMsg (str): the error message for the MCP server.
            metadata (dict): the metadata for the MCP server.
    """

    def __init__(self, client, mcpServerConnectionId=None, createdAt=None, updatedAt=None, name=None, config=None, description=None, transport=None, authType=None, externalConnectionId=None, inactive=None, tools=None, errorMsg=None, metadata=None):
        super().__init__(client, mcpServerConnectionId)
        self.mcp_server_connection_id = mcpServerConnectionId
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.name = name
        self.config = config
        self.description = description
        self.transport = transport
        self.auth_type = authType
        self.external_connection_id = externalConnectionId
        self.inactive = inactive
        self.tools = tools
        self.error_msg = errorMsg
        self.metadata = metadata
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'mcp_server_connection_id': repr(self.mcp_server_connection_id), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at), f'name': repr(self.name), f'config': repr(self.config), f'description': repr(self.description), f'transport': repr(
            self.transport), f'auth_type': repr(self.auth_type), f'external_connection_id': repr(self.external_connection_id), f'inactive': repr(self.inactive), f'tools': repr(self.tools), f'error_msg': repr(self.error_msg), f'metadata': repr(self.metadata)}
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
        resp = {'mcp_server_connection_id': self.mcp_server_connection_id, 'created_at': self.created_at, 'updated_at': self.updated_at, 'name': self.name, 'config': self.config, 'description': self.description,
                'transport': self.transport, 'auth_type': self.auth_type, 'external_connection_id': self.external_connection_id, 'inactive': self.inactive, 'tools': self.tools, 'error_msg': self.error_msg, 'metadata': self.metadata}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
