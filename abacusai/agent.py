from .agent_version import AgentVersion
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
            latestAgentVersion (AgentVersion): The latest agent version.
            codeSource (CodeSource): If a python model, information on the source code
    """

    def __init__(self, client, name=None, agentId=None, createdAt=None, projectId=None, notebookId=None, predictFunctionName=None, sourceCode=None, agentConfig=None, memory=None, codeSource={}, latestAgentVersion={}):
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
        self.code_source = client._build_class(CodeSource, codeSource)
        self.latest_agent_version = client._build_class(
            AgentVersion, latestAgentVersion)

    def __repr__(self):
        return f"Agent(name={repr(self.name)},\n  agent_id={repr(self.agent_id)},\n  created_at={repr(self.created_at)},\n  project_id={repr(self.project_id)},\n  notebook_id={repr(self.notebook_id)},\n  predict_function_name={repr(self.predict_function_name)},\n  source_code={repr(self.source_code)},\n  agent_config={repr(self.agent_config)},\n  memory={repr(self.memory)},\n  code_source={repr(self.code_source)},\n  latest_agent_version={repr(self.latest_agent_version)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'agent_id': self.agent_id, 'created_at': self.created_at, 'project_id': self.project_id, 'notebook_id': self.notebook_id, 'predict_function_name': self.predict_function_name, 'source_code': self.source_code, 'agent_config': self.agent_config, 'memory': self.memory, 'code_source': self._get_attribute_as_dict(self.code_source), 'latest_agent_version': self._get_attribute_as_dict(self.latest_agent_version)}

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

    def wait_for_publish(self, timeout=None):
        """
        A waiting call until agent is published.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
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
