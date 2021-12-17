from .return_class import AbstractApiClass


class RefreshPolicy(AbstractApiClass):
    """
        A Refresh Policy describes the frequency in which one or more datasets/models/deployments/batch_predictions can be updated.
    """

    def __init__(self, client, refreshPolicyId=None, name=None, cron=None, nextRunTime=None, createdAt=None, refreshType=None, projectId=None, datasetIds=None, modelIds=None, deploymentIds=None, paused=None):
        super().__init__(client, refreshPolicyId)
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
        return f"RefreshPolicy(refresh_policy_id={repr(self.refresh_policy_id)},\n  name={repr(self.name)},\n  cron={repr(self.cron)},\n  next_run_time={repr(self.next_run_time)},\n  created_at={repr(self.created_at)},\n  refresh_type={repr(self.refresh_type)},\n  project_id={repr(self.project_id)},\n  dataset_ids={repr(self.dataset_ids)},\n  model_ids={repr(self.model_ids)},\n  deployment_ids={repr(self.deployment_ids)},\n  paused={repr(self.paused)})"

    def to_dict(self):
        return {'refresh_policy_id': self.refresh_policy_id, 'name': self.name, 'cron': self.cron, 'next_run_time': self.next_run_time, 'created_at': self.created_at, 'refresh_type': self.refresh_type, 'project_id': self.project_id, 'dataset_ids': self.dataset_ids, 'model_ids': self.model_ids, 'deployment_ids': self.deployment_ids, 'paused': self.paused}

    def delete(self):
        """Delete a refresh policy"""
        return self.client.delete_refresh_policy(self.refresh_policy_id)

    def refresh(self):
        """Calls describe and refreshes the current object's fields"""
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """Retrieve a single refresh policy"""
        return self.client.describe_refresh_policy(self.refresh_policy_id)

    def list_refresh_pipeline_runs(self):
        """List the the times that the refresh policy has been run"""
        return self.client.list_refresh_pipeline_runs(self.refresh_policy_id)

    def pause(self):
        """Pauses a refresh policy"""
        return self.client.pause_refresh_policy(self.refresh_policy_id)

    def resume(self):
        """Resumes a refresh policy"""
        return self.client.resume_refresh_policy(self.refresh_policy_id)

    def run(self):
        """Force a run of the refresh policy."""
        return self.client.run_refresh_policy(self.refresh_policy_id)

    def update(self, name=None, cron=None):
        """Update the name or cron string of a  refresh policy"""
        return self.client.update_refresh_policy(self.refresh_policy_id, name, cron)
