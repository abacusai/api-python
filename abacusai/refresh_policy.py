

class RefreshPolicy():
    '''
        A Refresh Policy describes the frequency in which one or more datasets/models/deployments/batch_predictions can be updated.
    '''

    def __init__(self, client, refreshPolicyId=None, name=None, cron=None, nextRunTime=None, createdAt=None, refreshType=None, projectId=None, datasetIds=None, modelIds=None, deploymentIds=None, paused=None):
        self.client = client
        self.id = refreshPolicyId
        self.refresh_policy_id = refreshPolicyId
        self.name = name
        self.cron = cron
        self.next_run_time = nextRunTime
        self.created_at = createdAt
        self.refresh_type = refreshType
        self.project_id = projectId
        self.dataset_ids = datasetIds
        self.model_ids = modelIds
        self.deployment_ids = deploymentIds
        self.paused = paused

    def __repr__(self):
        return f"RefreshPolicy(refresh_policy_id={repr(self.refresh_policy_id)}, name={repr(self.name)}, cron={repr(self.cron)}, next_run_time={repr(self.next_run_time)}, created_at={repr(self.created_at)}, refresh_type={repr(self.refresh_type)}, project_id={repr(self.project_id)}, dataset_ids={repr(self.dataset_ids)}, model_ids={repr(self.model_ids)}, deployment_ids={repr(self.deployment_ids)}, paused={repr(self.paused)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'refresh_policy_id': self.refresh_policy_id, 'name': self.name, 'cron': self.cron, 'next_run_time': self.next_run_time, 'created_at': self.created_at, 'refresh_type': self.refresh_type, 'project_id': self.project_id, 'dataset_ids': self.dataset_ids, 'model_ids': self.model_ids, 'deployment_ids': self.deployment_ids, 'paused': self.paused}
