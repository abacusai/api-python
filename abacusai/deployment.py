

class Deployment():
    '''

    '''

    def __init__(self, client, deploymentId=None, name=None, status=None, description=None, deployedAt=None, createdAt=None, projectId=None, modelId=None, modelVersion=None, refreshSchedules=None, batchPredictionRefreshSchedules=None, callsPerSecond=None):
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
        self.refresh_schedules = refreshSchedules
        self.batch_prediction_refresh_schedules = batchPredictionRefreshSchedules
        self.calls_per_second = callsPerSecond

    def __repr__(self):
        return f"Deployment(deployment_id={repr(self.deployment_id)}, name={repr(self.name)}, status={repr(self.status)}, description={repr(self.description)}, deployed_at={repr(self.deployed_at)}, created_at={repr(self.created_at)}, project_id={repr(self.project_id)}, model_id={repr(self.model_id)}, model_version={repr(self.model_version)}, refresh_schedules={repr(self.refresh_schedules)}, batch_prediction_refresh_schedules={repr(self.batch_prediction_refresh_schedules)}, calls_per_second={repr(self.calls_per_second)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'deployment_id': self.deployment_id, 'name': self.name, 'status': self.status, 'description': self.description, 'deployed_at': self.deployed_at, 'created_at': self.created_at, 'project_id': self.project_id, 'model_id': self.model_id, 'model_version': self.model_version, 'refresh_schedules': self.refresh_schedules, 'batch_prediction_refresh_schedules': self.batch_prediction_refresh_schedules, 'calls_per_second': self.calls_per_second}

    def refresh(self):
        self = self.describe()
        return self

    def describe(self):
        return self.client.describe_deployment(self.deployment_id)

    def update(self, name=None, description=None):
        return self.client.update_deployment(self.deployment_id, name, description)

    def start(self):
        return self.client.start_deployment(self.deployment_id)

    def set_model_version(self, model_version):
        return self.client.set_deployment_model_version(self.deployment_id, model_version)

    def stop(self):
        return self.client.stop_deployment(self.deployment_id)

    def delete(self):
        return self.client.delete_deployment(self.deployment_id)

    def create_batch_prediction(self, name=None, global_prediction_args=None, explanations=False, output_location=None, database_connector_id=None, object_name=None, id_column=None, value_column=None, percentage_column=None, explanation_url_column=None):
        return self.client.create_batch_prediction(self.deployment_id, name, global_prediction_args, explanations, output_location, database_connector_id, object_name, id_column, value_column, percentage_column, explanation_url_column)

    def list_batch_predictions(self):
        return self.client.list_batch_predictions(self.deployment_id)

    def wait_for_deployment(self, timeout=480):
        return self.client._poll(self, {'PENDING', 'DEPLOYING'}, timeout=timeout)

    def get_status(self):
        return self.describe().status
