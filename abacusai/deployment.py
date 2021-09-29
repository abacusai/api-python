from .refresh_schedule import RefreshSchedule


class Deployment():
    '''
        A model deployment
    '''

    def __init__(self, client, deploymentId=None, name=None, status=None, description=None, deployedAt=None, createdAt=None, projectId=None, modelId=None, modelVersion=None, featureGroupId=None, featureGroupVersion=None, callsPerSecond=None, autoDeploy=None, regions=None, error=None, refreshSchedules={}):
        self.client = client
        self.id = deploymentId
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
        return f"Deployment(deployment_id={repr(self.deployment_id)}, name={repr(self.name)}, status={repr(self.status)}, description={repr(self.description)}, deployed_at={repr(self.deployed_at)}, created_at={repr(self.created_at)}, project_id={repr(self.project_id)}, model_id={repr(self.model_id)}, model_version={repr(self.model_version)}, feature_group_id={repr(self.feature_group_id)}, feature_group_version={repr(self.feature_group_version)}, calls_per_second={repr(self.calls_per_second)}, auto_deploy={repr(self.auto_deploy)}, regions={repr(self.regions)}, error={repr(self.error)}, refresh_schedules={repr(self.refresh_schedules)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'deployment_id': self.deployment_id, 'name': self.name, 'status': self.status, 'description': self.description, 'deployed_at': self.deployed_at, 'created_at': self.created_at, 'project_id': self.project_id, 'model_id': self.model_id, 'model_version': self.model_version, 'feature_group_id': self.feature_group_id, 'feature_group_version': self.feature_group_version, 'calls_per_second': self.calls_per_second, 'auto_deploy': self.auto_deploy, 'regions': self.regions, 'error': self.error, 'refresh_schedules': self.refresh_schedules.to_dict() if self.refresh_schedules else None}

    def refresh(self):
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        return self.client.describe_deployment(self.deployment_id)

    def update(self, description=None):
        return self.client.update_deployment(self.deployment_id, description)

    def rename(self, name):
        return self.client.rename_deployment(self.deployment_id, name)

    def set_auto(self, enable=None):
        return self.client.set_auto_deployment(self.deployment_id, enable)

    def set_model_version(self, model_version):
        return self.client.set_deployment_model_version(self.deployment_id, model_version)

    def set_feature_group_version(self, feature_group_version):
        return self.client.set_deployment_feature_group_version(self.deployment_id, feature_group_version)

    def start(self):
        return self.client.start_deployment(self.deployment_id)

    def stop(self):
        return self.client.stop_deployment(self.deployment_id)

    def delete(self):
        return self.client.delete_deployment(self.deployment_id)

    def create_batch_prediction(self, name=None, global_prediction_args=None, explanations=False, output_format=None, output_location=None, database_connector_id=None, database_output_config=None, refresh_schedule=None, csv_input_prefix=None, csv_prediction_prefix=None, csv_explanations_prefix=None):
        return self.client.create_batch_prediction(self.deployment_id, name, global_prediction_args, explanations, output_format, output_location, database_connector_id, database_output_config, refresh_schedule, csv_input_prefix, csv_prediction_prefix, csv_explanations_prefix)

    def wait_for_deployment(self, wait_states={'PENDING', 'DEPLOYING'}, timeout=480):
        return self.client._poll(self, wait_states, timeout=timeout)

    def get_status(self):
        return self.describe().status
