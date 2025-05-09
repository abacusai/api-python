import ast
import dataclasses
from typing import Dict, List, Tuple, Union

from . import enums
from .abstract import ApiClass, get_clean_function_source_code_for_agent, validate_constructor_arg_types


def validate_input_dict_param(dict_object, friendly_class_name, must_contain=[]):
    if not isinstance(dict_object, dict):
        raise ValueError(friendly_class_name, 'Invalid argument. Provided argument should be a dictionary.')
    if any(field not in dict_object for field in must_contain):
        raise ValueError(friendly_class_name, f'One or more keys are missing in the argument provided. Must contain keys - {must_contain}.')


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
class JSONSchema:
    @classmethod
    def from_fields_list(cls, fields_list: List[str]):
        if not fields_list:
            return cls(json_schema={})
        json_schema = {
            'type': 'object',
            'properties': {field: {'title': field, 'type': 'string'} for field in fields_list}
        }
        return cls(json_schema=json_schema)

    @classmethod
    def to_fields_list(cls, json_schema) -> List[str]:
        if 'properties' in json_schema:
            return list(json_schema['properties'].keys())
        return []


@validate_constructor_arg_types('input_mapping')
@dataclasses.dataclass
class WorkflowNodeInputMapping(ApiClass):
    """
    Represents a mapping of inputs to a workflow node.

    Args:
        name (str): The name of the input variable of the node function.
        variable_type (Union[WorkflowNodeInputType, str]): The type of the input. If the type is `IGNORE`, the input will be ignored.
        variable_source (str): The name of the node this variable is sourced from.
                               If the type is `WORKFLOW_VARIABLE`, the value given by the source node will be directly used.
                               If the type is `USER_INPUT`, the value given by the source node will be used as the default initial value before the user edits it.
                               Set to `None` if the type is `USER_INPUT` and the variable doesn't need a pre-filled initial value.
        is_required (bool): Indicates whether the input is required. Defaults to True.
        description (str): The description of this input.
        constant_value (str): The constant value of this input if variable type is CONSTANT. Only applicable for template nodes.
    """
    name: str
    variable_type: enums.WorkflowNodeInputType
    variable_source: str = dataclasses.field(default=None)
    source_prop: str = dataclasses.field(default=None)
    is_required: bool = dataclasses.field(default=True)
    description: str = dataclasses.field(default=None)
    constant_value: str = dataclasses.field(default=None)

    def __post_init__(self):
        if self.variable_type == enums.WorkflowNodeInputType.IGNORE:
            if self.is_required:
                raise ValueError('input_mapping', 'Invalid input mapping. The variable type cannot be IGNORE if is_required is True.')
            if self.variable_source or self.source_prop:
                raise ValueError('input_mapping', 'variable source and source prop should not be provided for IGNORE input mappings.')
        if self.variable_type != enums.WorkflowNodeInputType.CONSTANT and self.constant_value:
            raise ValueError('input_mapping', 'Invalid input mapping. If the variable type is not CONSTANT, constant_value must be empty.')
        if self.variable_type == enums.WorkflowNodeInputType.CONSTANT:
            if self.is_required and self.constant_value is None:
                raise ValueError('input_mapping', 'The constant value mapping should be provided for required CONSTANT input mappings.')
            if self.variable_source or self.source_prop:
                raise ValueError('input_mapping', 'variable source and source prop should not be provided for CONSTANT input mappings.')
        if isinstance(self.variable_type, str):
            self.variable_type = enums.WorkflowNodeInputType(self.variable_type)

    def to_dict(self):
        return {
            'name': self.name,
            'variable_type': self.variable_type.value,
            'variable_source': self.variable_source,
            'source_prop': self.source_prop or self.name,
            'is_required': self.is_required,
            'description': self.description,
            'constant_value': self.constant_value
        }

    @classmethod
    def from_dict(cls, mapping: dict):
        validate_input_dict_param(mapping, friendly_class_name='input_mapping', must_contain=['name', 'variable_type'])
        if not isinstance(mapping['variable_type'], str) and not isinstance(mapping['variable_type'], enums.WorkflowNodeInputType):
            raise ValueError('input_mapping', 'Invalid variable_type. Provided argument should be of type str or WorkflowNodeInputType enum.')
        if mapping['variable_type'] not in enums.WorkflowNodeInputType.__members__:
            raise ValueError('input_mapping', f"Invalid enum argument {mapping['variable_type']}. Provided argument should be of enum type WorkflowNodeInputType.")
        return cls(
            name=mapping['name'],
            variable_type=enums.WorkflowNodeInputType(mapping['variable_type']),
            variable_source=mapping.get('variable_source'),
            source_prop=mapping.get('source_prop') or mapping['name'] if mapping.get('variable_source') else None,
            is_required=mapping.get('is_required', True),
            description=mapping.get('description'),
            constant_value=mapping.get('constant_value')
        )


@validate_constructor_arg_types('input_schema')
@dataclasses.dataclass
class WorkflowNodeInputSchema(ApiClass, JSONSchema):
    """
    A schema conformant to react-jsonschema-form for workflow node input.

    To initialize a WorkflowNodeInputSchema dependent on another node's output, use the from_workflow_node method.

    Args:
        json_schema (dict): The JSON schema for the input, conformant to react-jsonschema-form specification. Must define keys like "title", "type", and "properties". Supported elements include Checkbox, Radio Button, Dropdown, Textarea, Number, Date, and file upload. Nested elements, arrays, and other complex types are not supported.
        ui_schema (dict): The UI schema for the input, conformant to react-jsonschema-form specification.
    """
    json_schema: dict
    ui_schema: dict = dataclasses.field(default_factory=dict)
    schema_source: str = dataclasses.field(default=None, init=False)
    schema_prop: str = dataclasses.field(default=None, init=False)
    runtime_schema: bool = dataclasses.field(default=False, init=False)

    def to_dict(self):
        if self.runtime_schema:
            return {
                'schema_source': self.schema_source,
                'schema_prop': self.schema_prop,
                'runtime_schema': self.runtime_schema
            }
        else:
            return {
                'json_schema': self.json_schema,
                'ui_schema': self.ui_schema,
                'runtime_schema': self.runtime_schema
            }

    @classmethod
    def from_dict(cls, schema: dict):
        validate_input_dict_param(schema, friendly_class_name='input_schema')
        if schema.get('runtime_schema', False):
            return cls.from_workflow_node(
                schema_source=schema.get('schema_source'),
                schema_prop=schema.get('schema_prop')
            )
        else:
            return cls(
                json_schema=schema.get('json_schema', schema),
                ui_schema=schema.get('ui_schema', {})
            )

    @classmethod
    def from_workflow_node(cls, schema_source: str, schema_prop: str):
        """
        Creates a WorkflowNodeInputSchema instance which references the schema generated by a WorkflowGraphNode.

        Args:
            schema_source (str): The name of the source WorkflowGraphNode.
            schema_prop (str): The name of the input schema parameter which source node outputs.
        """
        if not schema_source or not schema_prop:
            raise ValueError('input_schema', 'Valid schema_source and schema_prop must be provided for runtime schema.')
        if not isinstance(schema_source, str) or not isinstance(schema_prop, str):
            raise ValueError('input_schema', 'schema_source and schema_prop must be strings.')
        instance = cls(json_schema={})
        instance.schema_source = schema_source
        instance.schema_prop = schema_prop
        instance.runtime_schema = True
        return instance

    @classmethod
    def from_input_mappings(cls, input_mappings: List[WorkflowNodeInputMapping]):
        """
        Creates a json_schema for the input schema of the node from it's input mappings.

        Args:
            input_mappings (List[WorkflowNodeInputMapping]): The input mappings for the node.
        """
        user_input_mappings = [input_mapping for input_mapping in input_mappings if input_mapping.variable_type == enums.WorkflowNodeInputType.USER_INPUT]
        if len(user_input_mappings) > 0:
            json_schema = {
                'type': 'object',
                'required': [input_mapping.name for input_mapping in user_input_mappings if input_mapping.is_required],
                'properties': {input_mapping.name: {'title': input_mapping.name, 'type': 'string'} for input_mapping in user_input_mappings}
            }
            return cls(json_schema=json_schema)
        else:
            return cls(json_schema={})

    @classmethod
    def from_tool_variable_mappings(cls, tool_variable_mappings: dict):
        """
        Creates a WorkflowNodeInputSchema for the given tool variable mappings.

        Args:
            tool_variable_mappings (List[dict]): The tool variable mappings for the node.
        """
        json_schema = {'type': 'object', 'properties': {}}
        for mapping in tool_variable_mappings:
            if not mapping.get('is_required'):
                continue
            json_schema['properties'][mapping['name']] = {'title': mapping['name'], 'type': enums.PythonFunctionArgumentType.to_json_type(mapping['variable_type'])}
            if mapping['variable_type'] == enums.PythonFunctionArgumentType.ATTACHMENT:
                json_schema['properties'][mapping['name']]['format'] = 'data-url'
            if mapping['variable_type'] == enums.PythonFunctionArgumentType.LIST:
                json_schema['properties'][mapping['name']]['items'] = {'type': enums.PythonFunctionArgumentType.to_json_type(mapping['item_type'])}
                if mapping['item_type'] == enums.PythonFunctionArgumentType.ATTACHMENT:
                    json_schema['properties'][mapping['name']]['items']['format'] = 'data-url'
        return cls(json_schema=json_schema)


@validate_constructor_arg_types('output_mapping')
@dataclasses.dataclass
class WorkflowNodeOutputMapping(ApiClass):
    """
    Represents a mapping of output from a workflow node.

    Args:
        name (str): The name of the output.
        variable_type (Union[WorkflowNodeOutputType, str]): The type of the output in the form of an enum or a string.
        description (str): The description of this output.
    """
    name: str
    variable_type: Union[enums.WorkflowNodeOutputType, str] = dataclasses.field(default=enums.WorkflowNodeOutputType.ANY)
    description: str = dataclasses.field(default=None)

    def __post_init__(self):
        if isinstance(self.variable_type, str):
            self.variable_type = enums.WorkflowNodeOutputType(self.variable_type)

    def to_dict(self):
        return {
            'name': self.name,
            'variable_type': self.variable_type.value,
            'description': self.description
        }

    @classmethod
    def from_dict(cls, mapping: dict):
        validate_input_dict_param(mapping, friendly_class_name='output_mapping', must_contain=['name'])
        variable_type = mapping.get('variable_type', 'ANY')
        if not isinstance(variable_type, str) and not isinstance(variable_type, enums.WorkflowNodeOutputType):
            raise ValueError('output_mapping', 'Invalid variable_type. Provided argument should be of type str or WorkflowNodeOutputType enum.')
        if variable_type not in enums.WorkflowNodeOutputType.__members__:
            raise ValueError('output_mapping', f'Invalid enum argument {variable_type}. Provided argument should be of enum type WorkflowNodeOutputType.')
        return cls(
            name=mapping['name'],
            variable_type=enums.WorkflowNodeOutputType(variable_type),
            description=mapping.get('description')
        )


@validate_constructor_arg_types('output_schema')
@dataclasses.dataclass
class WorkflowNodeOutputSchema(ApiClass, JSONSchema):
    """
    A schema conformant to react-jsonschema-form for a workflow node output.

    Args:
        json_schema (dict): The JSON schema for the output, conformant to react-jsonschema-form specification.
    """
    json_schema: dict

    def to_dict(self):
        return {
            'json_schema': self.json_schema
        }

    @classmethod
    def from_dict(cls, schema: dict):
        validate_input_dict_param(schema, friendly_class_name='output_schema')
        return cls(
            json_schema=schema.get('json_schema', schema)
        )


@validate_constructor_arg_types('trigger_config')
@dataclasses.dataclass
class TriggerConfig(ApiClass):
    """
    Represents the configuration for a trigger workflow node.

    Args:
        sleep_time (int): The time in seconds to wait before the node gets executed again.
    """
    sleep_time: int = dataclasses.field(default=None)

    def to_dict(self):
        return {
            'sleep_time': self.sleep_time
        }

    @classmethod
    def from_dict(cls, configs: dict):
        validate_input_dict_param(configs, friendly_class_name='trigger_config')
        return cls(
            sleep_time=configs.get('sleep_time', None)
        )


@validate_constructor_arg_types('workflow_graph_node')
@dataclasses.dataclass
class WorkflowGraphNode(ApiClass):
    """
    Represents a node in an Agent workflow graph.

    Args:
        name (str): A unique name for the workflow node.
        input_mappings (List[WorkflowNodeInputMapping]): List of input mappings for the node. Each arg/kwarg of the node function should have a corresponding input mapping.
        output_mappings (List[str]): List of outputs for the node. Each field in the returned dict/AgentResponse must have a corresponding output in the list.
        function (callable): The callable node function reference.
        input_schema (WorkflowNodeInputSchema): The react json schema for the user input variables.
        output_schema (List[str]): The list of outputs to be shown on UI. Each output corresponds to a field in the output mappings of the node.

    Additional Attributes:
        function_name (str): The name of the function.
        source_code (str): The source code of the function.
        trigger_config (TriggerConfig): The configuration for a trigger workflow node.
    """

    def __init__(self, name: str, function: callable = None, input_mappings: Union[Dict[str, WorkflowNodeInputMapping], List[WorkflowNodeInputMapping]] = None, output_mappings: Union[List[str], Dict[str, str], List[WorkflowNodeOutputMapping]] = None, function_name: str = None, source_code: str = None, input_schema: Union[List[str], WorkflowNodeInputSchema] = None, output_schema: Union[List[str], WorkflowNodeOutputSchema] = None, template_metadata: dict = None, trigger_config: TriggerConfig = None):
        self.template_metadata = template_metadata
        self.trigger_config = trigger_config
        self.node_type = 'workflow_node'
        if self.template_metadata and not self.template_metadata.get('initialized'):
            self.name = name
            self.function_name = None
            self.source_code = None
            self.input_mappings = input_mappings
            self.output_mappings = []
            self.input_schema = input_schema
            self.output_schema = output_schema
        else:
            if function:
                self.function = function
                self.function_name = function.__name__
                self.source_code = get_clean_function_source_code_for_agent(function)
            elif function_name and source_code:
                self.function_name = function_name
                self.source_code = source_code
            else:
                raise ValueError('workflow_graph_node', 'Either function or function_name and source_code must be provided.')

            self.name = name
            try:
                tree = ast.parse(self.source_code)
            except SyntaxError as e:
                raise ValueError(f'"{name}" source code', f'SyntaxError: "{e}"')
            arg_defaults = {}
            function_found = False
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef) and node.name == self.function_name:
                    function_found = True
                    input_arguments = [arg.arg for arg in node.args.args]
                    defaults = [None] * (len(input_arguments) - len(node.args.defaults)) + node.args.defaults
                    arg_defaults = dict(zip(input_arguments, defaults))
            if not function_found:
                raise ValueError(f'"{name}" source code', f'Function "{self.function_name}" not found in the provided source code.')

            is_shortform_input_mappings = False
            if input_mappings is None:
                input_mappings = {}
            if isinstance(input_mappings, List) and all(isinstance(input, WorkflowNodeInputMapping) for input in input_mappings):
                self.input_mappings = input_mappings
                input_mapping_args = [input.name for input in input_mappings]
                for input_name in input_mapping_args:
                    if input_name not in arg_defaults:
                        raise ValueError('workflow_graph_node', f'Invalid input mapping. Argument "{input_name}" not found in function "{self.function_name}".')
                for arg, default in arg_defaults.items():
                    if arg not in input_mapping_args:
                        self.input_mappings.append(WorkflowNodeInputMapping(name=arg, variable_type=enums.WorkflowNodeInputType.USER_INPUT, is_required=default is None))
            elif isinstance(input_mappings, Dict) and all(isinstance(key, str) and isinstance(value, (WorkflowNodeInputMapping, WorkflowGraphNode)) for key, value in input_mappings.items()):
                is_shortform_input_mappings = True
                self.input_mappings = [WorkflowNodeInputMapping(name=arg, variable_type=enums.WorkflowNodeInputType.USER_INPUT, is_required=default is None) for arg, default in arg_defaults.items() if arg not in input_mappings]
                for key, value in input_mappings.items():
                    if key not in arg_defaults:
                        raise ValueError('workflow_graph_node', f'Invalid input mapping. Argument "{key}" not found in function "{self.function_name}".')
                    if isinstance(value, WorkflowGraphNode):
                        self.input_mappings.append(WorkflowNodeInputMapping(name=key, variable_type=enums.WorkflowNodeInputType.WORKFLOW_VARIABLE, variable_source=value.name, source_prop=key, is_required=arg_defaults.get(key) is None))
                    else:
                        self.input_mappings.append(WorkflowNodeInputMapping(name=key, variable_type=value.variable_type, variable_source=value.variable_source, source_prop=value.source_prop, is_required=arg_defaults.get(key) is None))
            else:
                raise ValueError('workflow_graph_node', 'Invalid input mappings. Must be a list of WorkflowNodeInputMapping or a dictionary of input mappings in the form {arg_name: node_name.outputs.prop_name}.')

            if input_schema is None:
                self.input_schema = WorkflowNodeInputSchema.from_input_mappings(self.input_mappings)
            elif isinstance(input_schema, WorkflowNodeInputSchema):
                self.input_schema = input_schema
            elif isinstance(input_schema, list) and all(isinstance(field, str) for field in input_schema):
                self.input_schema = WorkflowNodeInputSchema.from_fields_list(input_schema)
            else:
                raise ValueError('workflow_graph_node', 'Invalid input schema. Must be a WorkflowNodeInputSchema or a list of field names.')

            if input_schema is not None and is_shortform_input_mappings:
                # If user provided input_schema and input_mappings in shortform, then we need to update the input_mappings to have the correct variable_type
                user_input_fields = JSONSchema.to_fields_list(self.input_schema.json_schema)
                for mapping in self.input_mappings:
                    if mapping.name in user_input_fields:
                        mapping.variable_type = enums.WorkflowNodeInputType.USER_INPUT

            if output_mappings is None:
                output_mappings = []
            if isinstance(output_mappings, List):
                if all(isinstance(output, WorkflowNodeOutputMapping) for output in output_mappings):
                    self.output_mappings = output_mappings
                elif all(isinstance(output, str) for output in output_mappings):
                    self.output_mappings = [WorkflowNodeOutputMapping(name=output, variable_type=enums.WorkflowNodeOutputType.STRING) for output in output_mappings]
                else:
                    raise ValueError('workflow_graph_node', 'Invalid output mappings. Must be a list of WorkflowNodeOutputMapping or a list of output names or a dictionary of output mappings in the form {output_name: output_type}.')
            elif isinstance(output_mappings, Dict):
                self.output_mappings = [WorkflowNodeOutputMapping(name=output, variable_type=enums.WorkflowNodeOutputType.normalize_type(type)) for output, type in output_mappings.items()]
            else:
                raise ValueError('workflow_graph_node', 'Invalid output mappings. Must be a list of WorkflowNodeOutputMapping or a list of output names or a dictionary of output mappings in the form {output_name: output_type}.')

            if output_schema is None:
                outputs = [output.name for output in self.output_mappings]
                self.output_schema = WorkflowNodeOutputSchema.from_fields_list(outputs)
            elif isinstance(output_schema, WorkflowNodeOutputSchema):
                self.output_schema = output_schema
            elif isinstance(output_schema, list) and all(isinstance(field, str) for field in output_schema):
                self.output_schema = WorkflowNodeOutputSchema.from_fields_list(output_schema)
            else:
                raise ValueError('workflow_graph_node', 'Invalid output schema. Must be a WorkflowNodeOutputSchema or a list of output section names.')

    @classmethod
    def _raw_init(cls, name: str, input_mappings: List[WorkflowNodeInputMapping] = None, output_mappings: List[WorkflowNodeOutputMapping] = None, function: callable = None, function_name: str = None, source_code: str = None, input_schema: WorkflowNodeInputSchema = None, output_schema: WorkflowNodeOutputSchema = None, template_metadata: dict = None, trigger_config: TriggerConfig = None):
        workflow_node = cls.__new__(cls, name, input_mappings, output_mappings, input_schema, output_schema, template_metadata, trigger_config)
        workflow_node.name = name
        if function:
            workflow_node.function = function
            workflow_node.function_name = function.__name__
            workflow_node.source_code = get_clean_function_source_code_for_agent(function)
        elif function_name and (source_code or template_metadata):
            workflow_node.function_name = function_name
            workflow_node.source_code = source_code
        elif template_metadata and not template_metadata.get('initialized'):
            workflow_node.function_name = function_name
            workflow_node.source_code = source_code
        else:
            raise ValueError('workflow_graph_node', 'Either function or function_name and source_code must be provided.')
        workflow_node.input_mappings = input_mappings
        workflow_node.output_mappings = output_mappings
        workflow_node.input_schema = input_schema
        workflow_node.output_schema = output_schema
        workflow_node.template_metadata = template_metadata
        workflow_node.trigger_config = trigger_config
        return workflow_node

    @classmethod
    def from_template(cls, template_name: str, name: str, configs: dict = None, input_mappings: Union[Dict[str, WorkflowNodeInputMapping], List[WorkflowNodeInputMapping]] = None, input_schema: Union[List[str], WorkflowNodeInputSchema] = None, output_schema: Union[List[str], WorkflowNodeOutputSchema] = None, sleep_time: int = None):

        instance_input_mappings = []
        if isinstance(input_mappings, List) and all(isinstance(input, WorkflowNodeInputMapping) for input in input_mappings):
            instance_input_mappings = input_mappings
        elif isinstance(input_mappings, Dict) and all(isinstance(key, str) and isinstance(value, WorkflowNodeInputMapping) for key, value in input_mappings.items()):
            instance_input_mappings = [WorkflowNodeInputMapping(name=arg, variable_type=mapping.variable_type, variable_source=mapping.variable_source, source_prop=mapping.source_prop, is_required=mapping.is_required, description=mapping.description) for arg, mapping in input_mappings]
        elif input_mappings is None:
            instance_input_mappings = []
        else:
            raise ValueError('workflow_graph_node', 'Invalid input mappings. Must be a list of WorkflowNodeInputMapping or a dictionary of input mappings in the form {arg_name: node_name.outputs.prop_name}.')

        instance_input_schema = None
        if input_schema is None:
            instance_input_schema = WorkflowNodeInputSchema(json_schema={}, ui_schema={})
        elif isinstance(input_schema, WorkflowNodeInputSchema):
            instance_input_schema = input_schema
        elif isinstance(input_schema, list) and all(isinstance(field, str) for field in input_schema):
            instance_input_schema = WorkflowNodeInputSchema.from_fields_list(input_schema)
        else:
            raise ValueError('workflow_graph_node', 'Invalid input schema. Must be a WorkflowNodeInputSchema or a list of field names.')

        instance_output_schema = None
        if output_schema is None:
            instance_output_schema = WorkflowNodeOutputSchema(json_schema={})
        elif isinstance(output_schema, WorkflowNodeOutputSchema):
            instance_output_schema = output_schema
        elif isinstance(output_schema, list) and all(isinstance(field, str) for field in output_schema):
            instance_output_schema = WorkflowNodeOutputSchema.from_fields_list(output_schema)
        else:
            raise ValueError('workflow_graph_node', 'Invalid output schema. Must be a WorkflowNodeOutputSchema or a list of output section names.')

        if sleep_time is not None:
            if isinstance(sleep_time, str) and sleep_time.isdigit():
                sleep_time = int(sleep_time)
            if not isinstance(sleep_time, int) or sleep_time < 0:
                raise ValueError('workflow_graph_node', 'Invalid sleep time. Must be a non-negative integer.')

        return cls(
            name=name,
            input_mappings=instance_input_mappings,
            input_schema=instance_input_schema,
            output_schema=instance_output_schema,
            template_metadata={
                'template_name': template_name,
                'configs': configs or {},
                'initialized': False,
                'sleep_time': sleep_time,
            }
        )

    @classmethod
    def from_tool(cls, tool_name: str, name: str, configs: dict = None, input_mappings: Union[Dict[str, WorkflowNodeInputMapping], List[WorkflowNodeInputMapping]] = None, input_schema: Union[List[str], WorkflowNodeInputSchema] = None, output_schema: Union[List[str], WorkflowNodeOutputSchema] = None):
        """
        Creates and returns a WorkflowGraphNode based on an available user created tool.
        Note: DO NOT specify the output mapping for the tool; it will be inferred automatically. Doing so will raise an error.

        Args:
            tool_name: The name of the tool. There should already be a tool created in the platform with tool_name.
            name: The name to assign to the WorkflowGraphNode instance.
            configs (optional): The configuration state of the tool to use (if necessary). If not specified, will use the tool's default configuration.
            input_mappings (optional): The WorkflowNodeInputMappings for this node.
            input_schema (optional): The WorkflowNodeInputSchema for this node.
            output_schema (optional): The WorkflowNodeOutputSchema for this node.
        """
        node = cls.from_template(
            template_name=tool_name,
            name=name,
            configs=configs,
            input_mappings=input_mappings,
            input_schema=input_schema,
            output_schema=output_schema
        )
        node.template_metadata['template_type'] = 'tool'
        return node

    @classmethod
    def from_system_tool(cls, tool_name: str, name: str, configs: dict = None, input_mappings: Union[Dict[str, WorkflowNodeInputMapping], List[WorkflowNodeInputMapping]] = None, input_schema: Union[List[str], WorkflowNodeInputSchema] = None, output_schema: Union[List[str], WorkflowNodeOutputSchema] = None):
        """
        Creates and returns a WorkflowGraphNode based on the name of an available system tool.
        Note: DO NOT specify the output mapping for the tool; it will be inferred automatically. Doing so will raise an error.

        Args:
            tool_name: The name of the tool. There should already be a tool created in the platform with tool_name.
            name: The name to assign to the WorkflowGraphNode instance.
            configs (optional): The configuration state of the tool to use (if necessary). If not specified, will use the tool's default configuration.
            input_mappings (optional): The WorkflowNodeInputMappings for this node.
            input_schema (optional): The WorkflowNodeInputSchema for this node.
            output_schema (optional): The WorkflowNodeOutputSchema for this node.
        """
        node = cls.from_tool(tool_name, name, configs, input_mappings, input_schema, output_schema)
        node.template_metadata['is_system_tool'] = True
        return node

    def to_dict(self):
        return {
            'name': self.name,
            'function_name': self.function_name,
            'source_code': self.source_code,
            'input_mappings': [mapping.to_dict() for mapping in self.input_mappings],
            'output_mappings': [mapping.to_dict() for mapping in self.output_mappings],
            'input_schema': self.input_schema.to_dict(),
            'output_schema': self.output_schema.to_dict(),
            'template_metadata': self.template_metadata,
            'trigger_config': self.trigger_config.to_dict() if self.trigger_config else None,
            'node_type': self.node_type,
        }

    def is_template_node(self):
        return self.template_metadata is not None

    def is_trigger_node(self):
        return self.trigger_config is not None

    @classmethod
    def from_dict(cls, node: dict):
        if node.get('template_metadata'):
            node['function_name'] = node.get('function_name')
            node['source_code'] = node.get('source_code')

        validate_input_dict_param(node, friendly_class_name='workflow_graph_node', must_contain=['name', 'function_name', 'source_code'])
        _cls = cls._raw_init if node.get('__return_filter') else cls
        if node.get('template_metadata') and node.get('template_metadata').get('template_type') == 'trigger':
            if not node.get('trigger_config'):
                node['trigger_config'] = {'sleep_time': node.get('template_metadata').get('sleep_time')}
        instance = _cls(
            name=node['name'],
            function_name=node['function_name'],
            source_code=node['source_code'],
            input_mappings=[WorkflowNodeInputMapping.from_dict(mapping) for mapping in node.get('input_mappings', [])],
            output_mappings=[WorkflowNodeOutputMapping.from_dict(mapping) for mapping in node.get('output_mappings', [])],
            input_schema=WorkflowNodeInputSchema.from_dict(node.get('input_schema', {})),
            output_schema=WorkflowNodeOutputSchema.from_dict(node.get('output_schema', {})),
            template_metadata=node.get('template_metadata'),
            trigger_config=TriggerConfig.from_dict(node.get('trigger_config')) if node.get('trigger_config') else None
        )
        return instance

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == 'function':
            if value:
                self.function_name = value.__name__
                self.source_code = get_clean_function_source_code_for_agent(value)

    def __getattribute__(self, name):
        if name == 'function':
            try:
                val = super().__getattribute__(name)
            except AttributeError:
                val = None
            if val is None and self.function_name and self.source_code:
                raise AttributeError("This WorkflowGraphNode object was not created using a callable `function`. Please refer to `function_name` and `source_code` attributes to get it's function's details.")
            return val
        return super().__getattribute__(name)

    class Outputs:
        def __init__(self, node: 'WorkflowGraphNode'):
            self.node = node

        def __getattr__(self, name):
            for mapping in self.node.output_mappings:
                if mapping.name == name:
                    return WorkflowNodeInputMapping(name, enums.WorkflowNodeInputType.WORKFLOW_VARIABLE, variable_source=self.node.name, source_prop=name)
            raise AttributeError(f'Output mapping "{name}" not found in node "{self.name}".')

    @property
    def outputs(self):
        return self.Outputs(self)


@validate_constructor_arg_types('decision_node')
@dataclasses.dataclass
class DecisionNode(WorkflowGraphNode):
    """
    Represents a decision node in an Agent workflow graph. It is connected between two workflow nodes and is used to determine if subsequent nodes should be executed.
    """

    def __init__(self, name: str, condition: str, input_mappings: Union[Dict[str, WorkflowNodeInputMapping], List[WorkflowNodeInputMapping]]):
        self.node_type = 'decision_node'
        self.name = name
        self.source_code = condition
        if isinstance(input_mappings, List) and all(isinstance(input, WorkflowNodeInputMapping) for input in input_mappings):
            self.input_mappings = input_mappings
        elif isinstance(input_mappings, Dict) and all(isinstance(key, str) and isinstance(value, (WorkflowNodeInputMapping, WorkflowGraphNode)) for key, value in input_mappings.items()):
            self.input_mappings = []
            for key, value in input_mappings.items():
                variable_source = value.name if isinstance(value, WorkflowGraphNode) else value.variable_source
                source_prop = key if isinstance(value, WorkflowGraphNode) else value.source_prop
                self.input_mappings.append(WorkflowNodeInputMapping(name=key, variable_type=value.variable_type, variable_source=variable_source, source_prop=source_prop, is_required=value.is_required))
        else:
            raise ValueError('workflow_graph_decision_node', 'Invalid input mappings. Must be a list of WorkflowNodeInputMapping or a dictionary of input mappings in the form {arg_name: node_name.outputs.prop_name}.')
        for mapping in self.input_mappings:
            if mapping.variable_type != enums.WorkflowNodeInputType.WORKFLOW_VARIABLE:
                raise ValueError('workflow_graph_decision_node', 'Invalid input mappings. Decision node input mappings must be of type WORKFLOW_VARIABLE.')
            if not mapping.is_required:
                raise ValueError('workflow_graph_decision_node', 'Invalid input mappings. Decision node input mappings must be required.')
        self.output_mappings = [WorkflowNodeOutputMapping(name=input_mapping.name) for input_mapping in self.input_mappings]
        self.template_metadata = None
        self.trigger_config = None
        self.input_schema = None
        self.output_schema = None
        self.function_name = None

    def to_dict(self):
        return {
            'name': self.name,
            'source_code': self.source_code,
            'input_mappings': [mapping.to_dict() for mapping in self.input_mappings],
            'output_mappings': [mapping.to_dict() for mapping in self.output_mappings],
            'node_type': self.node_type,
        }

    @classmethod
    def from_dict(cls, node: dict):
        return cls(
            name=node['name'],
            condition=node['source_code'],
            input_mappings=[WorkflowNodeInputMapping.from_dict(mapping) for mapping in node.get('input_mappings', [])]
        )


@validate_constructor_arg_types('workflow_graph_edge')
@dataclasses.dataclass
class WorkflowGraphEdge(ApiClass):
    """
    Represents an edge in an Agent workflow graph.

    To make an edge conditional, provide {'EXECUTION_CONDITION': '<condition>'} key-value in the details dictionary.
    The condition should be a Pythonic expression string that evaluates to a boolean value and only depends on the outputs of the source node of the edge.

    Args:
        source (str): The name of the source node of the edge.
        target (str): The name of the target node of the edge.
        details (dict): Additional details about the edge. Like the condition for edge execution.
    """
    source: Union[str, WorkflowGraphNode]
    target: Union[str, WorkflowGraphNode]
    details: dict = dataclasses.field(default_factory=dict)

    def __init__(self, source: Union[str, WorkflowGraphNode], target: Union[str, WorkflowGraphNode], details: dict = None):
        self.source = source.name if isinstance(source, WorkflowGraphNode) else source
        self.target = target.name if isinstance(target, WorkflowGraphNode) else target
        self.details = details or {}

    def to_nx_edge(self):
        return [self.source, self.target, self.details]


@validate_constructor_arg_types('workflow_graph')
@dataclasses.dataclass
class WorkflowGraph(ApiClass):
    """
    Represents an Agent workflow graph.

    Args:
        nodes (List[Union[WorkflowGraphNode, DecisionNode]]): A list of nodes in the workflow graph.
        primary_start_node (Union[str, WorkflowGraphNode]): The primary node to start the workflow from.
        common_source_code (str): Common source code that can be used across all nodes.
    """
    nodes: List[Union[WorkflowGraphNode, DecisionNode]] = dataclasses.field(default_factory=list)
    edges: List[Union[WorkflowGraphEdge, Tuple[WorkflowGraphNode, WorkflowGraphNode, dict], Tuple[str, str, dict]]] = dataclasses.field(default_factory=list, metadata={'deprecated': True})
    primary_start_node: Union[str, WorkflowGraphNode] = dataclasses.field(default=None)
    common_source_code: str = dataclasses.field(default=None)
    specification_type: str = dataclasses.field(default='data_flow')

    def __post_init__(self):
        if self.specification_type == 'execution_flow' and any(isinstance(node, DecisionNode) for node in self.nodes):
            raise ValueError('workflow_graph', 'Decision nodes are not supported in execution flow specification type.')
        if self.edges:
            if self.specification_type == 'execution_flow':
                for index, edge in enumerate(self.edges):
                    if isinstance(edge, Tuple):
                        source = edge[0] if isinstance(edge[0], str) else edge[0].name
                        target = edge[1] if isinstance(edge[1], str) else edge[1].name
                        details = edge[2] if len(edge) > 2 and isinstance(edge[2], dict) else None
                        self.edges[index] = WorkflowGraphEdge(source=source, target=target, details=details)
            else:
                raise ValueError('workflow_graph', 'Workflow Graph no longer supports explicit edges. They are inferred from data flow dependencies.')

    def to_dict(self):
        return {
            'nodes': [node.to_dict() for node in self.nodes],
            **({'edges': [edge.to_dict() for edge in self.edges]} if self.specification_type == 'execution_flow' else {}),
            'primary_start_node': self.primary_start_node.name if isinstance(self.primary_start_node, WorkflowGraphNode) else self.primary_start_node,
            'common_source_code': self.common_source_code,
            'specification_type': self.specification_type
        }

    @classmethod
    def from_dict(cls, graph: dict):
        validate_input_dict_param(graph, friendly_class_name='workflow_graph')
        if graph.get('__return_filter'):
            for node in graph.get('nodes', []):
                node['__return_filter'] = True
        nodes = [DecisionNode.from_dict(node) if node.get('node_type') == 'decision_node' else WorkflowGraphNode.from_dict(node) for node in graph.get('nodes', [])]
        edges = [WorkflowGraphEdge.from_dict(edge) for edge in graph.get('edges', [])]
        primary_start_node = graph.get('primary_start_node')
        non_primary_nodes = set()
        specification_type = graph.get('specification_type', 'execution_flow')

        if specification_type == 'execution_flow':
            if primary_start_node not in [node.name for node in nodes]:
                for edge in edges:
                    non_primary_nodes.add(edge.target)
                primary_nodes = set([node.name for node in nodes]) - non_primary_nodes
                graph['primary_start_node'] = primary_nodes.pop() if primary_nodes else None

            return cls(
                nodes=nodes,
                edges=edges,
                primary_start_node=graph.get('primary_start_node', None),
                common_source_code=graph.get('common_source_code', None),
                specification_type='execution_flow'
            )
        else:
            if edges:
                raise ValueError('workflow_graph', 'Workflow Graph no longer supports explicit edges. They are inferred from data flow dependencies.')

            if primary_start_node not in [node.name for node in nodes]:
                for node in nodes:
                    is_primary_eligible = True
                    for mapping in node.input_mappings:
                        if mapping.variable_type == enums.WorkflowNodeInputType.WORKFLOW_VARIABLE:
                            is_primary_eligible = False
                            break
                    if not is_primary_eligible:
                        non_primary_nodes.add(node.name)
                primary_nodes = set([node.name for node in nodes]) - non_primary_nodes
                primary_start_node = primary_nodes.pop() if primary_nodes else None
            return cls(
                nodes=nodes,
                primary_start_node=primary_start_node,
                common_source_code=graph.get('common_source_code', None),
                specification_type='data_flow'
            )


@dataclasses.dataclass
class AgentConversationMessage(ApiClass):
    """
    Message format for agent conversation

    Args:
        is_user (bool): Whether the message is from the user.
        text (str): The message's text.
        document_contents (dict): Dict of document name to document text in case of any document present.
    """
    is_user: bool = dataclasses.field(default=None)
    text: str = dataclasses.field(default=None)
    document_contents: dict = dataclasses.field(default=None)

    def to_dict(self):
        return {
            'is_user': self.is_user,
            'text': self.text,
            'document_contents': self.document_contents
        }


@dataclasses.dataclass
class WorkflowNodeTemplateConfig(ApiClass):
    """
    Represents a WorkflowNode template config.

    Args:
        name (str): A unique name of the config.
        description (str): The description of this config.
        default_value (str): Default value of the config to be used if value is not provided during node initialization.
        is_required (bool): Indicates whether the config is required. Defaults to False.
    """
    name: str
    description: str = dataclasses.field(default=None)
    default_value: str = dataclasses.field(default=None)
    is_required: bool = dataclasses.field(default=False)

    def to_dict(self):
        return {
            'name': self.name,
            'description': self.description,
            'default_value': self.default_value,
            'is_required': self.is_required
        }

    @classmethod
    def from_dict(cls, mapping: dict):
        return cls(
            name=mapping['name'],
            description=mapping.get('description'),
            default_value=mapping.get('default_value'),
            is_required=mapping.get('is_required', False)
        )


@dataclasses.dataclass
class WorkflowNodeTemplateInput(ApiClass):
    """
    Represents an input to the workflow node generated using template.

    Args:
        name (str): A unique name of the input.
        is_required (bool): Indicates whether the input is required. Defaults to False.
        description (str): The description of this input.
    """
    name: str
    is_required: bool = dataclasses.field(default=False)
    description: str = dataclasses.field(default='')

    def to_dict(self):
        return {
            'name': self.name,
            'is_required': self.is_required,
            'description': self.description
        }

    @classmethod
    def from_dict(cls, mapping: dict):
        return cls(
            name=mapping['name'],
            is_required=mapping.get('is_required', False),
            description=mapping.get('description', '')
        )


@dataclasses.dataclass
class WorkflowNodeTemplateOutput(ApiClass):
    """
    Represents an output returned by the workflow node generated using template.

    Args:
        name (str): The name of the output.
        variable_type (WorkflowNodeOutputType): The type of the output.
        description (str): The description of this output.
    """
    name: str
    variable_type: enums.WorkflowNodeOutputType = dataclasses.field(default=enums.WorkflowNodeOutputType.ANY)
    description: str = dataclasses.field(default='')

    def to_dict(self):
        return {
            'name': self.name,
            'variable_type': self.variable_type.value,
            'description': self.description
        }

    @classmethod
    def from_dict(cls, mapping: dict):
        return cls(
            name=mapping['name'],
            variable_type=enums.WorkflowNodeOutputType(mapping.get('variable_type', 'ANY')),
            description=mapping.get('description', '')
        )
