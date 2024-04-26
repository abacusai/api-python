import dataclasses
from typing import List, Union

from . import enums
from .abstract import ApiClass, get_clean_function_source_code


@dataclasses.dataclass
class FieldDescriptor(ApiClass):
    """
    Configs for vector store indexing.

    Args:
        field (str): The field to be extracted. This will be used as the key in the response.
        description (str): The description of this field. If not included, the response_field will be used.
        example_extraction (Union[str, int, bool, float]): An example of this extracted field.
        type (FieldDescriptorType): The type of this field. If not provided, the default type is STRING.
    """
    field: str = dataclasses.field()
    description: str = dataclasses.field(default=None)
    example_extraction: Union[str, int, bool, float, list, dict] = dataclasses.field(default=None)
    type: enums.FieldDescriptorType = dataclasses.field(default=enums.FieldDescriptorType.STRING)


@dataclasses.dataclass
class WorkflowNodeInputMapping(ApiClass):
    """
    A mapping of input to a workflow node.

    Args:
        name (str): The name of the input.
        variable_type (WorkflowNodeInputType): The type of the input.
        variable_source (str): The name of the node this variable is sourced from.
                               If the type is `WORKFLOW_VARIABLE`, the value given by the source node will be directly used.
                               If the type is `USER_INPUT`, the value given by the source node will be used as the default initial value before user edits it.
                               Set to `None` if the type is `USER_INPUT` and the variable doesn't need a pre-filled initial value.
        is_required (bool): Whether the input is required.
    """
    name: str
    variable_type: enums.WorkflowNodeInputType
    variable_source: str = dataclasses.field(default=None)
    is_required: bool = dataclasses.field(default=True)

    def to_dict(self):
        return {
            'name': self.name,
            'variable_type': self.variable_type,
            'variable_source': self.variable_source,
            'is_required': self.is_required
        }


@dataclasses.dataclass
class WorkflowNodeOutputMapping(ApiClass):
    """
    A mapping of output to a workflow node.

    Args:
        name (str): The name of the output.
        variable_type (WorkflowNodeOutputType): The type of the output.
    """
    name: str
    variable_type: enums.WorkflowNodeOutputType = dataclasses.field(default=enums.WorkflowNodeOutputType.STRING)

    def to_dict(self):
        return {
            'name': self.name,
            'variable_type': self.variable_type
        }


@dataclasses.dataclass
class WorkflowGraphNode(ApiClass):
    """
    A node in an Agent workflow graph.

    Args:
        name (str): A unique name for the workflow node.
        input_mappings (List[WorkflowNodeInputMapping]): List of input mappings for the node.
        output_mappings (List[WorkflowNodeOutputMapping]): List of output mappings for the node.
        function (callable): The callable node function reference.
        input_schema (dict): The react json schema for the input form if applicable.
        output_schema (dict): The react json schema for the output if applicable.
        package_requirements (list): List of package requirements for the node.
    """

    def __init__(self, name: str, input_mappings: List[WorkflowNodeInputMapping], output_mappings: List[WorkflowNodeOutputMapping], function: callable = None, function_name: str = None, source_code: str = None, input_schema: dict = None, output_schema: dict = None, package_requirements: list = None):
        if function:
            self.function_name = function.__name__
            self.source_code = get_clean_function_source_code(function)
        elif function_name and source_code:
            self.function_name = function_name
            self.source_code = source_code
        else:
            raise ValueError('Either function or function_name and source_code must be provided.')

        self.name = name
        self.input_mappings = input_mappings
        self.output_mappings = output_mappings
        self.input_schema = input_schema if input_schema else {}
        self.output_schema = output_schema if output_schema else {}
        self.package_requirements = package_requirements if package_requirements else []

    def to_dict(self):
        return {
            'name': self.name,
            'function_name': self.function_name,
            'source_code': self.source_code,
            'input_mappings': [mapping.to_dict() for mapping in self.input_mappings],
            'output_mappings': [mapping.to_dict() for mapping in self.output_mappings],
            'input_schema': self.input_schema,
            'output_schema': self.output_schema,
            'package_requirements': self.package_requirements
        }

    @classmethod
    def from_dict(cls, node: dict):
        return cls(
            name=node['name'],
            function_name=node['function_name'],
            source_code=node['source_code'],
            input_mappings=[WorkflowNodeInputMapping(**mapping) for mapping in node['input_mappings']],
            output_mappings=[WorkflowNodeOutputMapping(**mapping) for mapping in node['output_mappings']],
            input_schema=node.get('input_schema', {}),
            output_schema=node.get('output_schema', {}),
            package_requirements=node.get('package_requirements', [])
        )


@dataclasses.dataclass
class WorkflowGraphEdge(ApiClass):
    """
    An edge in an Agent workflow graph.

    Args:
        source (str): The name of the source node of the edge.
        target (str): The name of the target node of the edge.
        details (dict): Additional details about the edge.
    """
    source: str
    target: str
    details: dict = dataclasses.field(default_factory=dict)

    def to_nx_edge(self):
        return [self.source, self.target, self.details]


@dataclasses.dataclass
class WorkflowGraph(ApiClass):
    """
    An Agent workflow graph.

    Args:
        nodes (List[WorkflowGraphNode]): A list of nodes in the workflow graph.
        edges (List[WorkflowGraphEdge]): A list of edges in the workflow graph, where each edge is a tuple of source, target and details.
    """
    nodes: List[WorkflowGraphNode] = dataclasses.field(default_factory=list)
    edges: List[WorkflowGraphEdge] = dataclasses.field(default_factory=list)

    def to_dict(self):
        return {
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges]
        }

    @classmethod
    def from_dict(cls, graph: dict):
        return cls(
            nodes=[WorkflowGraphNode.from_dict(node) for node in graph.get('nodes', [])],
            edges=[WorkflowGraphEdge.from_dict(edge) for edge in graph.get('edges', [])]
        )
