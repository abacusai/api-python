from .model_location import ModelLocation
from .model_version import ModelVersion
import time
from .refresh_schedule import RefreshSchedule


class Model():
    '''
        A model
    '''

    def __init__(self, client, name=None, modelId=None, modelConfig=None, createdAt=None, projectId=None, shared=None, sharedAt=None, trainFunctionName=None, predictFunctionName=None, trainingInputTables=None, sourceCode=None, location={}, refreshSchedules={}, latestModelVersion={}):
        self.client = client
        self.id = modelId
        self.name = name
        self.model_id = modelId
        self.model_config = modelConfig
        self.created_at = createdAt
        self.project_id = projectId
        self.shared = shared
        self.shared_at = sharedAt
        self.train_function_name = trainFunctionName
        self.predict_function_name = predictFunctionName
        self.training_input_tables = trainingInputTables
        self.source_code = sourceCode
        self.location = client._build_class(ModelLocation, location)
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)
        self.latest_model_version = client._build_class(
            ModelVersion, latestModelVersion)

    def __repr__(self):
        return f"Model(name={repr(self.name)}, model_id={repr(self.model_id)}, model_config={repr(self.model_config)}, created_at={repr(self.created_at)}, project_id={repr(self.project_id)}, shared={repr(self.shared)}, shared_at={repr(self.shared_at)}, train_function_name={repr(self.train_function_name)}, predict_function_name={repr(self.predict_function_name)}, training_input_tables={repr(self.training_input_tables)}, source_code={repr(self.source_code)}, location={repr(self.location)}, refresh_schedules={repr(self.refresh_schedules)}, latest_model_version={repr(self.latest_model_version)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'name': self.name, 'model_id': self.model_id, 'model_config': self.model_config, 'created_at': self.created_at, 'project_id': self.project_id, 'shared': self.shared, 'shared_at': self.shared_at, 'train_function_name': self.train_function_name, 'predict_function_name': self.predict_function_name, 'training_input_tables': self.training_input_tables, 'source_code': self.source_code, 'location': self.location.to_dict() if self.location else None, 'refresh_schedules': self.refresh_schedules.to_dict() if self.refresh_schedules else None, 'latest_model_version': self.latest_model_version.to_dict() if self.latest_model_version else None}

    def refresh(self):
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        return self.client.describe_model(self.model_id)

    def rename(self, name):
        return self.client.rename_model(self.model_id, name)

    def update_training_config(self, training_config):
        return self.client.update_model_training_config(self.model_id, training_config)

    def set_training_config(self, training_config):
        return self.client.set_model_training_config(self.model_id, training_config)

    def get_metrics(self, model_version=None, baseline_metrics=False):
        return self.client.get_model_metrics(self.model_id, model_version, baseline_metrics)

    def list_versions(self, limit=100, start_after_version=None):
        return self.client.list_model_versions(self.model_id, limit, start_after_version)

    def retrain(self, deployment_ids=[]):
        return self.client.retrain_model(self.model_id, deployment_ids)

    def delete(self):
        return self.client.delete_model(self.model_id)

    def wait_for_training(self, timeout=None):
        return self.client._poll(self, {'PENDING', 'TRAINING'}, delay=30, timeout=timeout)

    def wait_for_evaluation(self, timeout=None):
        return self.wait_for_training()

    def wait_for_full_automl(self, timeout=None):
        start_time = time.time()
        while True:
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f'Maximum wait time of {timeout}s exceeded')
            model_version = self.client._call_api('describeModel', 'GET', query_params={
                                                  'modelId': self.model_id, 'waitForFullAutoml': True}, parse_type=Model).latest_model_version
            if model_version.status not in {'PENDING', 'TRAINING'} and not model_version.pending_deployment_ids:
                break
            time.sleep(30)
        return self.describe()

    def get_status(self, get_automl_status: bool = False):
        if get_automl_status:
            return self.client._call_api('describeModel', 'GET', query_params={'modelId': self.model_id, 'waitForFullAutoml': True}, parse_type=Model).latest_model_version.status
        return self.describe().latest_model_version.status
