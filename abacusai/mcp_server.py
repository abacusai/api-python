from .return_class import AbstractApiClass


class McpServer(AbstractApiClass):
    """
        Model Context Protocol Server

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name of the MCP server.
            description (str): description of what the MCP server does.
            envVars (list): list of api_keys or credentials required by the MCP server.
            config (str): a json string containing the command and arguments for the MCP server.
            envVarInstructions (str): instructions for the user to get the environment variables.
            url (str): The url of the MCP server github repository or webpage.
            isActive (bool): Whether the MCP server is active.
            metadata (dict): additional information about the MCP server including github_stars, etc.
    """

    def __init__(self, client, name=None, description=None, envVars=None, config=None, envVarInstructions=None, url=None, isActive=None, metadata=None):
        super().__init__(client, None)
        self.name = name
        self.description = description
        self.env_vars = envVars
        self.config = config
        self.env_var_instructions = envVarInstructions
        self.url = url
        self.is_active = isActive
        self.metadata = metadata
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'description': repr(self.description), f'env_vars': repr(self.env_vars), f'config': repr(
            self.config), f'env_var_instructions': repr(self.env_var_instructions), f'url': repr(self.url), f'is_active': repr(self.is_active), f'metadata': repr(self.metadata)}
        class_name = "McpServer"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'description': self.description, 'env_vars': self.env_vars, 'config': self.config,
                'env_var_instructions': self.env_var_instructions, 'url': self.url, 'is_active': self.is_active, 'metadata': self.metadata}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
