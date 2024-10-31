from .api_class import WorkflowNodeTemplateConfig, WorkflowNodeTemplateInput, WorkflowNodeTemplateOutput
from .return_class import AbstractApiClass


class WorkflowNodeTemplate(AbstractApiClass):
    """
        A workflow node template.

        Args:
            client (ApiClient): An authenticated API Client instance
            workflowNodeTemplateId (str): The unique identifier of the workflow node template.
            name (str): The name of the workflow node template.
            functionName (str): The function name of the workflow node function.
            sourceCode (str): The source code of the function that the workflow node template will execute.
            description (str): A description of the workflow node template.
            packageRequirements (list[str]): A list of package requirements that the node source code may need.
            tags (dict): Tags to add to the workflow node template. It contains information on the intended usage of template.
            additionalConfigs (dict): Additional configurations for the workflow node template.
            inputs (WorkflowNodeTemplateInput): A list of inputs that the workflow node template will use.
            outputs (WorkflowNodeTemplateOutput): A list of outputs that the workflow node template will give.
            templateConfigs (WorkflowNodeTemplateConfig): A list of template configs that are hydrated into source to get complete code.
    """

    def __init__(self, client, workflowNodeTemplateId=None, name=None, functionName=None, sourceCode=None, description=None, packageRequirements=None, tags=None, additionalConfigs=None, inputs={}, outputs={}, templateConfigs={}):
        super().__init__(client, workflowNodeTemplateId)
        self.workflow_node_template_id = workflowNodeTemplateId
        self.name = name
        self.function_name = functionName
        self.source_code = sourceCode
        self.description = description
        self.package_requirements = packageRequirements
        self.tags = tags
        self.additional_configs = additionalConfigs
        self.inputs = client._build_class(WorkflowNodeTemplateInput, inputs)
        self.outputs = client._build_class(WorkflowNodeTemplateOutput, outputs)
        self.template_configs = client._build_class(
            WorkflowNodeTemplateConfig, templateConfigs)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'workflow_node_template_id': repr(self.workflow_node_template_id), f'name': repr(self.name), f'function_name': repr(self.function_name), f'source_code': repr(self.source_code), f'description': repr(self.description), f'package_requirements': repr(
            self.package_requirements), f'tags': repr(self.tags), f'additional_configs': repr(self.additional_configs), f'inputs': repr(self.inputs), f'outputs': repr(self.outputs), f'template_configs': repr(self.template_configs)}
        class_name = "WorkflowNodeTemplate"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'workflow_node_template_id': self.workflow_node_template_id, 'name': self.name, 'function_name': self.function_name, 'source_code': self.source_code, 'description': self.description, 'package_requirements': self.package_requirements,
                'tags': self.tags, 'additional_configs': self.additional_configs, 'inputs': self._get_attribute_as_dict(self.inputs), 'outputs': self._get_attribute_as_dict(self.outputs), 'template_configs': self._get_attribute_as_dict(self.template_configs)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
