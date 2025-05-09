from .return_class import AbstractApiClass


class McpConfig(AbstractApiClass):
    """
        Model Context Protocol Config

        Args:
            client (ApiClient): An authenticated API Client instance
            mcpConfig (dict): The MCP configuration for the current user
    """

    def __init__(self, client, mcpConfig=None):
        super().__init__(client, None)
        self.mcp_config = mcpConfig
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'mcp_config': repr(self.mcp_config)}
        class_name = "McpConfig"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'mcp_config': self.mcp_config}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
