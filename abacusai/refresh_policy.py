from .return_class import AbstractApiClass


class RefreshPolicy(AbstractApiClass):
    """
        A Refresh Policy describes the frequency in which one or more datasets/models/deployments/batch_predictions can be updated.

        Args:
            client (ApiClient): An authenticated API Client instance
            refreshPolicyId (str): The unique identifier for the refresh policy
            name (str): The user-friendly name for the refresh policy
            cron (str): A cron-style string that describes the when this refresh policy is to be executed in UTC
            nextRunTime (str): The next UTC time that this refresh policy will be executed
            createdAt (str): The time when the refresh policy was created
            refreshType (str): The type of refresh policy to be run
            projectId (str): The unique identifier of a project that this refresh policy applies to
            datasetIds (list of unique identifiers of type 'string'): Comma separated list of Dataset IDs that this refresh policy applies to
            modelIds (list of unique identifiers of type 'string'): Comma separated list of Model IDs that this refresh policy applies to
            deploymentIds (list of unique identifiers of type 'string'): Comma separated list of Deployment IDs that this refresh policy applies to
            predictionMetricIds (list of unique identifiers of type 'string'): Comma separated list of Prediction Metric IDs that this refresh policy applies to
            paused (bool): (Boolean): True if the refresh policy is paused
    """

    def __init__(self, client, refreshPolicyId=None, name=None, cron=None, nextRunTime=None, createdAt=None, refreshType=None, projectId=None, datasetIds=None, modelIds=None, deploymentIds=None, predictionMetricIds=None, paused=None):
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
        self.prediction_metric_ids = predictionMetricIds
        self.paused = paused

    def __repr__(self):
        return f"RefreshPolicy(refresh_policy_id={repr(self.refresh_policy_id)},\n  name={repr(self.name)},\n  cron={repr(self.cron)},\n  next_run_time={repr(self.next_run_time)},\n  created_at={repr(self.created_at)},\n  refresh_type={repr(self.refresh_type)},\n  project_id={repr(self.project_id)},\n  dataset_ids={repr(self.dataset_ids)},\n  model_ids={repr(self.model_ids)},\n  deployment_ids={repr(self.deployment_ids)},\n  prediction_metric_ids={repr(self.prediction_metric_ids)},\n  paused={repr(self.paused)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'refresh_policy_id': self.refresh_policy_id, 'name': self.name, 'cron': self.cron, 'next_run_time': self.next_run_time, 'created_at': self.created_at, 'refresh_type': self.refresh_type, 'project_id': self.project_id, 'dataset_ids': self.dataset_ids, 'model_ids': self.model_ids, 'deployment_ids': self.deployment_ids, 'prediction_metric_ids': self.prediction_metric_ids, 'paused': self.paused}

    def delete(self):
        """
        Delete a refresh policy

        Args:
            refresh_policy_id (str): The unique ID associated with this refresh policy
        """
        return self.client.delete_refresh_policy(self.refresh_policy_id)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            RefreshPolicy: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Retrieve a single refresh policy

        Args:
            refresh_policy_id (str): The unique ID associated with this refresh policy

        Returns:
            RefreshPolicy: A refresh policy object
        """
        return self.client.describe_refresh_policy(self.refresh_policy_id)

    def list_refresh_pipeline_runs(self):
        """
        List the the times that the refresh policy has been run

        Args:
            refresh_policy_id (str): The unique ID associated with this refresh policy

        Returns:
            RefreshPipelineRun: A list of refresh pipeline runs for the given refresh policy id
        """
        return self.client.list_refresh_pipeline_runs(self.refresh_policy_id)

    def pause(self):
        """
        Pauses a refresh policy

        Args:
            refresh_policy_id (str): The unique ID associated with this refresh policy
        """
        return self.client.pause_refresh_policy(self.refresh_policy_id)

    def resume(self):
        """
        Resumes a refresh policy

        Args:
            refresh_policy_id (str): The unique ID associated with this refresh policy
        """
        return self.client.resume_refresh_policy(self.refresh_policy_id)

    def run(self):
        """
        Force a run of the refresh policy.

        Args:
            refresh_policy_id (str): The unique ID associated with this refresh policy
        """
        return self.client.run_refresh_policy(self.refresh_policy_id)

    def update(self, name: str = None, cron: str = None):
        """
        Update the name or cron string of a  refresh policy

        Args:
            name (str): Optional, specify to update the name of the refresh policy
            cron (str): Optional, specify to update the cron string describing the schedule from the refresh policy

        Returns:
            RefreshPolicy: The updated refresh policy
        """
        return self.client.update_refresh_policy(self.refresh_policy_id, name, cron)
