from .api_class import WorkflowGraphNode
from .return_class import AbstractApiClass


class WorkflowGraphNodeDetails(AbstractApiClass):
    """
        A workflow graph node in the workflow graph.

        Args:
            client (ApiClient): An authenticated API Client instance
            workflowGraphNode (WorkflowGraphNode): The workflow graph node object.
    """

    def __init__(self, client, workflowGraphNode={}):
        super().__init__(client, None)
        self.workflow_graph_node = client._build_class(
            WorkflowGraphNode, workflowGraphNode)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'workflow_graph_node': repr(self.workflow_graph_node)}
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
        resp = {'workflow_graph_node': self._get_attribute_as_dict(
            self.workflow_graph_node)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}