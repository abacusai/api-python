from .refresh_schedule import RefreshSchedule
from .return_class import AbstractApiClass


class Deployment(AbstractApiClass):
    """
        A model deployment
    """

    def __init__(self, client, deploymentId=None, name=None, status=None, description=None, deployedAt=None, createdAt=None, projectId=None, modelId=None, modelVersion=None, featureGroupId=None, featureGroupVersion=None, callsPerSecond=None, autoDeploy=None, regions=None, error=None, refreshSchedules={}):
        super().__init__(client, deploymentId)
        self.deployment_id = deploymentId
        self.name = name
        self.status = status
        self.description = description
        self.deployed_at = deployedAt
        self.created_at = createdAt
        self.project_id = projectId
        self.model_id = modelId
        self.model_version = modelVersion
        self.feature_group_id = featureGroupId
        self.feature_group_version = featureGroupVersion
        self.calls_per_second = callsPerSecond
        self.auto_deploy = autoDeploy
        self.regions = regions
        self.error = error
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)

    def __repr__(self):
        return f"Deployment(deployment_id={repr(self.deployment_id)},\n  name={repr(self.name)},\n  status={repr(self.status)},\n  description={repr(self.description)},\n  deployed_at={repr(self.deployed_at)},\n  created_at={repr(self.created_at)},\n  project_id={repr(self.project_id)},\n  model_id={repr(self.model_id)},\n  model_version={repr(self.model_version)},\n  feature_group_id={repr(self.feature_group_id)},\n  feature_group_version={repr(self.feature_group_version)},\n  calls_per_second={repr(self.calls_per_second)},\n  auto_deploy={repr(self.auto_deploy)},\n  regions={repr(self.regions)},\n  error={repr(self.error)},\n  refresh_schedules={repr(self.refresh_schedules)})"

    def to_dict(self):
        return {'deployment_id': self.deployment_id, 'name': self.name, 'status': self.status, 'description': self.description, 'deployed_at': self.deployed_at, 'created_at': self.created_at, 'project_id': self.project_id, 'model_id': self.model_id, 'model_version': self.model_version, 'feature_group_id': self.feature_group_id, 'feature_group_version': self.feature_group_version, 'calls_per_second': self.calls_per_second, 'auto_deploy': self.auto_deploy, 'regions': self.regions, 'error': self.error, 'refresh_schedules': self._get_attribute_as_dict(self.refresh_schedules)}

    def refresh(self):
        """Calls describe and refreshes the current object's fields"""
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """Retrieves a full description of the specified deployment."""
        return self.client.describe_deployment(self.deployment_id)

    def update(self, description=None):
        """Updates a deployment's description."""
        return self.client.update_deployment(self.deployment_id, description)

    def rename(self, name):
        """Updates a deployment's name and/or description."""
        return self.client.rename_deployment(self.deployment_id, name)

    def set_auto(self, enable=None):
        """Enable/Disable auto deployment for the specified deployment."""
        return self.client.set_auto_deployment(self.deployment_id, enable)

    def set_model_version(self, model_version):
        """Promotes a Model Version to be served in the Deployment"""
        return self.client.set_deployment_model_version(self.deployment_id, model_version)

    def set_feature_group_version(self, feature_group_version):
        """Promotes a Feature Group Version to be served in the Deployment"""
        return self.client.set_deployment_feature_group_version(self.deployment_id, feature_group_version)

    def start(self):
        """Restarts the specified deployment that was previously suspended."""
        return self.client.start_deployment(self.deployment_id)

    def stop(self):
        """Stops the specified deployment."""
        return self.client.stop_deployment(self.deployment_id)

    def delete(self):
        """Deletes the specified deployment. The deployment's models will not be affected. Note that the deployments are not recoverable after they are deleted."""
        return self.client.delete_deployment(self.deployment_id)

    def create_batch_prediction(self, name=None, global_prediction_args=None, explanations=False, output_format=None, output_location=None, database_connector_id=None, database_output_config=None, refresh_schedule=None, csv_input_prefix=None, csv_prediction_prefix=None, csv_explanations_prefix=None):
        """Creates a batch prediction job description for the given deployment."""
        return self.client.create_batch_prediction(self.deployment_id, name, global_prediction_args, explanations, output_format, output_location, database_connector_id, database_output_config, refresh_schedule, csv_input_prefix, csv_prediction_prefix, csv_explanations_prefix)

    def wait_for_deployment(self, wait_states={'PENDING', 'DEPLOYING'}, timeout=480):
        """
        A waiting call until deployment is completed.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out. Default value given is 480 milliseconds.

        Returns:
            None
        """
        return self.client._poll(self, wait_states, timeout=timeout)

    def get_status(self):
        """
        Gets the status of the deployment.

        Returns:
            Enum (string): A string describing the status of a deploymet (pending, deploying, active, etc.).
        """
        return self.describe().status

    def create_refresh_policy(self, cron: str):
        """
        To create a refresh policy for a deployment.

        Args:
            cron (str): A cron style string to set the refresh time.

        Returns:
            RefreshPolicy (object): The refresh policy object.
        """
        return self.client.create_refresh_policy(self.name, cron, 'DEPLOYMENT', deployment_ids=[self.id])

    def list_refresh_policies(self):
        """
        Gets the refresh policies in a list.

        Returns:
            List (RefreshPolicy): A list of refresh policy objects.
        """
        return self.client.list_refresh_policies(deployment_ids=[self.id])
