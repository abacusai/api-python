from .return_class import AbstractApiClass


class FeatureGroupLineage(AbstractApiClass):
    """
        Directed acyclic graph of feature group lineage for all feature groups in a project

        Args:
            client (ApiClient): An authenticated API Client instance
            nodes (list<dict>): A list of nodes in the graph containing feature groups and datasets
            connections (list<dict>): A list of connections in the graph between nodes
    """

    def __init__(self, client, nodes=None, connections=None):
        super().__init__(client, None)
        self.nodes = nodes
        self.connections = connections
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'nodes': repr(self.nodes),
                     f'connections': repr(self.connections)}
        class_name = "FeatureGroupLineage"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'nodes': self.nodes, 'connections': self.connections}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
