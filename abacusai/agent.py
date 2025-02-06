from .agent_version import AgentVersion
from .api_class import WorkflowGraph
from .code_source import CodeSource
from .return_class import AbstractApiClass


class Agent(AbstractApiClass):
    """
        An AI agent.

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The user-friendly name for the agent.
            agentId (str): The unique identifier of the agent.
            createdAt (str): Date and time at which the agent was created.
            projectId (str): The project this agent belongs to.
            notebookId (str): The notebook associated with the agent.
            predictFunctionName (str): Name of the function found in the source code that will be executed run predictions through agent. It is not executed when this function is run.
            sourceCode (str): Python code used to make the agent.
            agentConfig (dict): The config options used to create this agent.
            memory (int): Memory in GB specified for the deployment resources for the agent.
            trainingRequired (bool): Whether training is required to deploy the latest agent code.
            agentExecutionConfig (dict): The config for arguments used to execute the agent.
            latestAgentVersion (AgentVersion): The latest agent version.
            codeSource (CodeSource): If a python model, information on the source code
            draftWorkflowGraph (WorkflowGraph): The saved draft state of the workflow graph for the agent.
            workflowGraph (WorkflowGraph): The workflow graph for the agent.
    """

    def __init__(self, client, name=None, agentId=None, createdAt=None, projectId=None, notebookId=None, predictFunctionName=None, sourceCode=None, agentConfig=None, memory=None, trainingRequired=None, agentExecutionConfig=None, codeSource={}, latestAgentVersion={}, draftWorkflowGraph={}, workflowGraph={}):
        super().__init__(client, agentId)
        self.name = name
        self.agent_id = agentId
        self.created_at = createdAt
        self.project_id = projectId
        self.notebook_id = notebookId
        self.predict_function_name = predictFunctionName
        self.source_code = sourceCode
        self.agent_config = agentConfig
        self.memory = memory
        self.training_required = trainingRequired
        self.agent_execution_config = agentExecutionConfig
        self.code_source = client._build_class(CodeSource, codeSource)
        self.latest_agent_version = client._build_class(
            AgentVersion, latestAgentVersion)
        self.draft_workflow_graph = client._build_class(
            WorkflowGraph, draftWorkflowGraph)
        self.workflow_graph = client._build_class(WorkflowGraph, workflowGraph)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'agent_id': repr(self.agent_id), f'created_at': repr(self.created_at), f'project_id': repr(self.project_id), f'notebook_id': repr(self.notebook_id), f'predict_function_name': repr(self.predict_function_name), f'source_code': repr(self.source_code), f'agent_config': repr(self.agent_config), f'memory': repr(
            self.memory), f'training_required': repr(self.training_required), f'agent_execution_config': repr(self.agent_execution_config), f'code_source': repr(self.code_source), f'latest_agent_version': repr(self.latest_agent_version), f'draft_workflow_graph': repr(self.draft_workflow_graph), f'workflow_graph': repr(self.workflow_graph)}
        class_name = "Agent"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'agent_id': self.agent_id, 'created_at': self.created_at, 'project_id': self.project_id, 'notebook_id': self.notebook_id, 'predict_function_name': self.predict_function_name, 'source_code': self.source_code, 'agent_config': self.agent_config, 'memory': self.memory, 'training_required': self.training_required,
                'agent_execution_config': self.agent_execution_config, 'code_source': self._get_attribute_as_dict(self.code_source), 'latest_agent_version': self._get_attribute_as_dict(self.latest_agent_version), 'draft_workflow_graph': self._get_attribute_as_dict(self.draft_workflow_graph), 'workflow_graph': self._get_attribute_as_dict(self.workflow_graph)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            Agent: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Retrieves a full description of the specified model.

        Args:
            agent_id (str): Unique string identifier associated with the model.

        Returns:
            Agent: Description of the agent.
        """
        return self.client.describe_agent(self.agent_id)

    def list_versions(self, limit: int = 100, start_after_version: str = None):
        """
        List all versions of an agent.

        Args:
            limit (int): If provided, limits the number of agent versions returned.
            start_after_version (str): Unique string identifier of the version after which the list starts.

        Returns:
            list[AgentVersion]: An array of Agent versions.
        """
        return self.client.list_agent_versions(self.agent_id, limit, start_after_version)

    def copy(self, project_id: str = None):
        """
        Creates a copy of the input agent

        Args:
            project_id (str): Project id to create the new agent to. By default it picks up the source agent's project id.

        Returns:
            Agent: The newly generated agent.
        """
        return self.client.copy_agent(self.agent_id, project_id)

    @property
    def description(self) -> str:
        """
        The description of the agent.
        """
        return (self.agent_config or {}).get('DESCRIPTION')

    @property
    def agent_interface(self) -> str:
        """
        The interface that the agent will be deployed with.
        """
        return (self.agent_config or {}).get('AGENT_INTERFACE')

    @property
    def agent_connectors(self) -> dict:
        """
        A dictionary mapping ApplicationConnectorType keys to lists of OAuth scopes. Each key represents a specific application connector, while the value is a list of scopes that define the permissions granted to the application.
        """
        return (self.agent_config or {}).get('AGENT_CONNECTORS')

    def wait_for_publish(self, timeout=None):
        """
        A waiting call until agent is published.

        Args:
            timeout (int): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        latest_agent_version = self.describe().latest_agent_version
        if not latest_agent_version:
            from .client import ApiException
            raise ApiException(409, 'This agent does not have any versions')
        self.latest_agent_version = latest_agent_version.wait_for_publish(
            timeout=timeout)
        return self

    def get_status(self):
        """
        Gets the status of the agent publishing.

        Returns:
            str: A string describing the status of a agent publishing (pending, complete, etc.).
        """
        return self.describe().latest_agent_version.status

    def republish(self):
        """
        Re-publishes the Agent and creates a new Agent Version.

        Returns:
            AgentVersion: The new Agent Version.
        """
        self.client.retrain_model(self.agent_id)
        return self.describe().latest_agent_version
