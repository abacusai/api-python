from .return_class import AbstractApiClass


class AuditLog(AbstractApiClass):
    """
        An audit log entry

        Args:
            client (ApiClient): An authenticated API Client instance
            createdAt (str): The timestamp when the audit log entry was created
            userId (str): The hashed ID of the user who performed the action
            objectId (str): The hashed ID of the object that was affected by the action
            action (str): The action performed (create, modify, start, stop, delete, share, hide, credential_change, login)
            source (str): The source of the action (api, ui, pipeline, cli, system)
            refreshPolicyId (str): The hashed ID of the refresh policy if applicable
            pipelineId (str): The hashed ID of the pipeline if applicable
    """

    def __init__(self, client, createdAt=None, userId=None, objectId=None, action=None, source=None, refreshPolicyId=None, pipelineId=None):
        super().__init__(client, None)
        self.created_at = createdAt
        self.user_id = userId
        self.object_id = objectId
        self.action = action
        self.source = source
        self.refresh_policy_id = refreshPolicyId
        self.pipeline_id = pipelineId
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'created_at': repr(self.created_at), f'user_id': repr(self.user_id), f'object_id': repr(self.object_id), f'action': repr(
            self.action), f'source': repr(self.source), f'refresh_policy_id': repr(self.refresh_policy_id), f'pipeline_id': repr(self.pipeline_id)}
        class_name = "AuditLog"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'created_at': self.created_at, 'user_id': self.user_id, 'object_id': self.object_id, 'action': self.action,
                'source': self.source, 'refresh_policy_id': self.refresh_policy_id, 'pipeline_id': self.pipeline_id}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
