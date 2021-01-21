from .model_version import ModelVersion


class Model():
    '''

    '''

    def __init__(self, client, name=None, modelId=None, modelConfig=None, createdAt=None, projectId=None, shared=None, sharedAt=None, refreshSchedules=None, location=None, latestModelVersion={}):
        self.client = client
        self.id = modelId
        self.name = name
        self.model_id = modelId
        self.model_config = modelConfig
        self.created_at = createdAt
        self.project_id = projectId
        self.shared = shared
        self.shared_at = sharedAt
        self.refresh_schedules = refreshSchedules
        self.location = location
        self.latest_model_version = client._build_class(
            ModelVersion, latestModelVersion)

    def __repr__(self):
        return f"Model(name={repr(self.name)}, model_id={repr(self.model_id)}, model_config={repr(self.model_config)}, created_at={repr(self.created_at)}, project_id={repr(self.project_id)}, shared={repr(self.shared)}, shared_at={repr(self.shared_at)}, refresh_schedules={repr(self.refresh_schedules)}, location={repr(self.location)}, latest_model_version={repr(self.latest_model_version)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'name': self.name, 'model_id': self.model_id, 'model_config': self.model_config, 'created_at': self.created_at, 'project_id': self.project_id, 'shared': self.shared, 'shared_at': self.shared_at, 'refresh_schedules': self.refresh_schedules, 'location': self.location, 'latest_model_version': self.latest_model_version.to_dict() if self.latest_model_version else None}

    def refresh(self):
        self = self.describe()
        return self

    def describe(self):
        return self.client.describe_model(self.model_id)

    def update_training_config(self, training_config):
        return self.client.update_model_training_config(self.model_id, training_config)

    def get_metrics(self, model_version=None, baseline_metrics=False):
        return self.client.get_model_metrics(self.model_id, model_version, baseline_metrics)

    def list_versions(self):
        return self.client.list_model_versions(self.model_id)

    def retrain(self, deployment_ids=[]):
        return self.client.retrain_model(self.model_id, deployment_ids)

    def cancel_training(self):
        return self.client.cancel_model_training(self.model_id)

    def delete(self):
        return self.client.delete_model(self.model_id)

    def create_deployment(self, name=None, description=None, calls_per_second=None):
        return self.client.create_deployment(self.model_id, name, description, calls_per_second)

    def wait_for_training(self, timeout=None):
        return self.client._poll(self, {'PENDING', 'TRAINING'}, delay=30, timeout=timeout)

    def wait_for_evaluation(self, timeout=None):
        return self.wait_for_training()

    def get_status(self):
        return self.describe().latest_model_version.status
