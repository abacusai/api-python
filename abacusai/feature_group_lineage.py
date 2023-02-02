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

    def __repr__(self):
        return f"FeatureGroupLineage(nodes={repr(self.nodes)},\n  connections={repr(self.connections)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'nodes': self.nodes, 'connections': self.connections}
