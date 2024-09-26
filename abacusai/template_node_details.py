from .api_class import WorkflowGraphNode
from .return_class import AbstractApiClass


class TemplateNodeDetails(AbstractApiClass):
    """
        Details about WorkflowGraphNode object and notebook code for adding template nodes in workflow.

        Args:
            client (ApiClient): An authenticated API Client instance
            notebookCode (list): The boilerplate code that needs to be shown in notebook for creating workflow graph node using corresponding template.
            workflowGraphNode (WorkflowGraphNode): The workflow graph node object corresponding to the template.
    """

    def __init__(self, client, notebookCode=None, workflowGraphNode={}):
        super().__init__(client, None)
        self.notebook_code = notebookCode
        self.workflow_graph_node = client._build_class(
            WorkflowGraphNode, workflowGraphNode)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'notebook_code': repr(
            self.notebook_code), f'workflow_graph_node': repr(self.workflow_graph_node)}
        class_name = "TemplateNodeDetails"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'notebook_code': self.notebook_code,
                'workflow_graph_node': self._get_attribute_as_dict(self.workflow_graph_node)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
