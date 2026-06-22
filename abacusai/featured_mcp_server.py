from .return_class import AbstractApiClass


class FeaturedMcpServer(AbstractApiClass):
    """
        A curated remote MCP server shown in the MCP server gallery.

        Args:
            client (ApiClient): An authenticated API Client instance
            key (str): Stable identifier; used as the connection name on connect.
            name (str): Display name of the MCP server.
            url (str): The remote MCP server (streamable HTTP) endpoint URL.
            tagline (str): One-line description shown on the gallery card.
            description (str): Longer description shown in the detail view.
            iconUrl (str): Full URL of the server icon.
            categories (list): Category labels for filtering.
            author (str): Name of the company/developer behind the server.
            authorUrl (str): Link to the author's site.
            docsUrl (str): Link to the server's documentation, if any.
            oauth2 (bool): Whether the server requires OAuth authentication.
            rank (int): Sort order; lower ranks are shown first.
            tools (list): Names of the tools the server provides.
            toolCount (int): Total tool count (differs from len(tools) only if the YAML caps the list).
    """

    def __init__(self, client, key=None, name=None, url=None, tagline=None, description=None, iconUrl=None, categories=None, author=None, authorUrl=None, docsUrl=None, oauth2=None, rank=None, tools=None, toolCount=None):
        super().__init__(client, None)
        self.key = key
        self.name = name
        self.url = url
        self.tagline = tagline
        self.description = description
        self.icon_url = iconUrl
        self.categories = categories
        self.author = author
        self.author_url = authorUrl
        self.docs_url = docsUrl
        self.oauth2 = oauth2
        self.rank = rank
        self.tools = tools
        self.tool_count = toolCount
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'key': repr(self.key), f'name': repr(self.name), f'url': repr(self.url), f'tagline': repr(self.tagline), f'description': repr(self.description), f'icon_url': repr(self.icon_url), f'categories': repr(
            self.categories), f'author': repr(self.author), f'author_url': repr(self.author_url), f'docs_url': repr(self.docs_url), f'oauth2': repr(self.oauth2), f'rank': repr(self.rank), f'tools': repr(self.tools), f'tool_count': repr(self.tool_count)}
        class_name = "FeaturedMcpServer"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'key': self.key, 'name': self.name, 'url': self.url, 'tagline': self.tagline, 'description': self.description, 'icon_url': self.icon_url, 'categories': self.categories,
                'author': self.author, 'author_url': self.author_url, 'docs_url': self.docs_url, 'oauth2': self.oauth2, 'rank': self.rank, 'tools': self.tools, 'tool_count': self.tool_count}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
