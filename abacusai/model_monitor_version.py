from .return_class import AbstractApiClass


class ModelMonitorVersion(AbstractApiClass):
    """
        A version of a model monitor
    """

    def __init__(self, client, modelMonitorVersion=None, status=None, modelMonitorId=None, monitoringStartedAt=None, monitoringCompletedAt=None, trainingFeatureGroupVersion=None, predictionFeatureGroupVersion=None, error=None, pendingDeploymentIds=None, failedDeploymentIds=None):
        super().__init__(client, modelMonitorVersion)
        self.model_monitor_version = modelMonitorVersion
        self.status = status
        self.model_monitor_id = modelMonitorId
        self.monitoring_started_at = monitoringStartedAt
        self.monitoring_completed_at = monitoringCompletedAt
        self.training_feature_group_version = trainingFeatureGroupVersion
        self.prediction_feature_group_version = predictionFeatureGroupVersion
        self.error = error
        self.pending_deployment_ids = pendingDeploymentIds
        self.failed_deployment_ids = failedDeploymentIds

    def __repr__(self):
        return f"ModelMonitorVersion(model_monitor_version={repr(self.model_monitor_version)},\n  status={repr(self.status)},\n  model_monitor_id={repr(self.model_monitor_id)},\n  monitoring_started_at={repr(self.monitoring_started_at)},\n  monitoring_completed_at={repr(self.monitoring_completed_at)},\n  training_feature_group_version={repr(self.training_feature_group_version)},\n  prediction_feature_group_version={repr(self.prediction_feature_group_version)},\n  error={repr(self.error)},\n  pending_deployment_ids={repr(self.pending_deployment_ids)},\n  failed_deployment_ids={repr(self.failed_deployment_ids)})"

    def to_dict(self):
        return {'model_monitor_version': self.model_monitor_version, 'status': self.status, 'model_monitor_id': self.model_monitor_id, 'monitoring_started_at': self.monitoring_started_at, 'monitoring_completed_at': self.monitoring_completed_at, 'training_feature_group_version': self.training_feature_group_version, 'prediction_feature_group_version': self.prediction_feature_group_version, 'error': self.error, 'pending_deployment_ids': self.pending_deployment_ids, 'failed_deployment_ids': self.failed_deployment_ids}

    def refresh(self):
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        return self.client.describe_model_monitor_version(self.model_monitor_version)

    def delete(self):
        return self.client.delete_model_monitor_version(self.model_monitor_version)

    def get_model_monitoring_logs(self, stdout=False, stderr=False):
        return self.client.get_model_monitoring_logs(self.model_monitor_version, stdout, stderr)

    def get_drift_for_feature(self, feature_name):
        return self.client.get_drift_for_feature(self.model_monitor_version, feature_name)
