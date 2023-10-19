from .code_source import CodeSource
from .return_class import AbstractApiClass


class AgentVersion(AbstractApiClass):
    """
        A version of an AI agent.

        Args:
            client (ApiClient): An authenticated API Client instance
            agentVersion (str): The unique identifier of an agent version.
            status (str): The current status of the model.
            agentId (str): A reference to the agent this version belongs to.
            agentConfig (dict): The config options used to create this agent.
            publishingStartedAt (str): The start time and date of the training process in ISO-8601 format.
            publishingCompletedAt (str): The end time and date of the training process in ISO-8601 format.
            pendingDeploymentIds (list): List of deployment IDs where deployment is pending.
            failedDeploymentIds (list): List of failed deployment IDs.
            error (str): Relevant error if the status is FAILED.
            codeSource (CodeSource): If a python model, information on where the source code is located.
    """

    def __init__(self, client, agentVersion=None, status=None, agentId=None, agentConfig=None, publishingStartedAt=None, publishingCompletedAt=None, pendingDeploymentIds=None, failedDeploymentIds=None, error=None, codeSource={}):
        super().__init__(client, agentVersion)
        self.agent_version = agentVersion
        self.status = status
        self.agent_id = agentId
        self.agent_config = agentConfig
        self.publishing_started_at = publishingStartedAt
        self.publishing_completed_at = publishingCompletedAt
        self.pending_deployment_ids = pendingDeploymentIds
        self.failed_deployment_ids = failedDeploymentIds
        self.error = error
        self.code_source = client._build_class(CodeSource, codeSource)

    def __repr__(self):
        repr_dict = {f'agent_version': repr(self.agent_version), f'status': repr(self.status), f'agent_id': repr(self.agent_id), f'agent_config': repr(self.agent_config), f'publishing_started_at': repr(self.publishing_started_at), f'publishing_completed_at': repr(
            self.publishing_completed_at), f'pending_deployment_ids': repr(self.pending_deployment_ids), f'failed_deployment_ids': repr(self.failed_deployment_ids), f'error': repr(self.error), f'code_source': repr(self.code_source)}
        class_name = "AgentVersion"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'agent_version': self.agent_version, 'status': self.status, 'agent_id': self.agent_id, 'agent_config': self.agent_config, 'publishing_started_at': self.publishing_started_at, 'publishing_completed_at': self.publishing_completed_at,
                'pending_deployment_ids': self.pending_deployment_ids, 'failed_deployment_ids': self.failed_deployment_ids, 'error': self.error, 'code_source': self._get_attribute_as_dict(self.code_source)}
        return {key: value for key, value in resp.items() if value is not None}

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            AgentVersion: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Retrieves a full description of the specified agent version.

        Args:
            agent_version (str): Unique string identifier of the agent version.

        Returns:
            AgentVersion: A agent version.
        """
        return self.client.describe_agent_version(self.agent_version)

    def wait_for_publish(self, timeout=None):
        """
        A waiting call until agent gets published.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'PUBLISHING'}, delay=30, timeout=timeout)

    def get_status(self):
        """
        Gets the status of the model version under training.

        Returns:
            str: A string describing the status of a model training (pending, complete, etc.).
        """
        return self.describe().status
