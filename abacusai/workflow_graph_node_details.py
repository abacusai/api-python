from .api_class import WorkflowGraphNode
from .return_class import AbstractApiClass


class WorkflowGraphNodeDetails(AbstractApiClass):
    """
        A workflow graph node in the workflow graph.

        Args:
            client (ApiClient): An authenticated API Client instance
            packageRequirements (list[str]): A list of package requirements that the node source code will need.
            connectors (dict): A dictionary of connectors that the node source code will need.
            workflowGraphNode (WorkflowGraphNode): The workflow graph node object.
    """

    def __init__(self, client, packageRequirements=None, connectors=None, workflowGraphNode={}):
        super().__init__(client, None)
        self.package_requirements = packageRequirements
        self.connectors = connectors
        self.workflow_graph_node = client._build_class(
            WorkflowGraphNode, workflowGraphNode)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'package_requirements': repr(self.package_requirements), f'connectors': repr(
            self.connectors), f'workflow_graph_node': repr(self.workflow_graph_node)}
        class_name = "WorkflowGraphNodeDetails"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'package_requirements': self.package_requirements, 'connectors': self.connectors,
                'workflow_graph_node': self._get_attribute_as_dict(self.workflow_graph_node)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
