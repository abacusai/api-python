from .batch_dataset import BatchDataset


class BatchPrediction():
    '''

    '''

    def __init__(self, client, batchPredictionId=None, name=None, status=None, deploymentId=None, inputLocation=None, outputLocation=None, predictionsStartedAt=None, predictionsCompletedAt=None, connectorOutputLocation=None, uploadId=None, globalPredictionArgs=None, totalPredictions=None, failedPredictions=None, inputDatasets={}):
        self.client = client
        self.id = batchPredictionId
        self.batch_prediction_id = batchPredictionId
        self.name = name
        self.status = status
        self.deployment_id = deploymentId
        self.input_location = inputLocation
        self.output_location = outputLocation
        self.predictions_started_at = predictionsStartedAt
        self.predictions_completed_at = predictionsCompletedAt
        self.connector_output_location = connectorOutputLocation
        self.upload_id = uploadId
        self.global_prediction_args = globalPredictionArgs
        self.total_predictions = totalPredictions
        self.failed_predictions = failedPredictions
        self.input_datasets = client._build_class(BatchDataset, inputDatasets)

    def __repr__(self):
        return f"BatchPrediction(batch_prediction_id={repr(self.batch_prediction_id)}, name={repr(self.name)}, status={repr(self.status)}, deployment_id={repr(self.deployment_id)}, input_location={repr(self.input_location)}, output_location={repr(self.output_location)}, predictions_started_at={repr(self.predictions_started_at)}, predictions_completed_at={repr(self.predictions_completed_at)}, connector_output_location={repr(self.connector_output_location)}, upload_id={repr(self.upload_id)}, global_prediction_args={repr(self.global_prediction_args)}, total_predictions={repr(self.total_predictions)}, failed_predictions={repr(self.failed_predictions)}, input_datasets={repr(self.input_datasets)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'batch_prediction_id': self.batch_prediction_id, 'name': self.name, 'status': self.status, 'deployment_id': self.deployment_id, 'input_location': self.input_location, 'output_location': self.output_location, 'predictions_started_at': self.predictions_started_at, 'predictions_completed_at': self.predictions_completed_at, 'connector_output_location': self.connector_output_location, 'upload_id': self.upload_id, 'global_prediction_args': self.global_prediction_args, 'total_predictions': self.total_predictions, 'failed_predictions': self.failed_predictions, 'input_datasets': self.input_datasets.to_dict() if self.input_datasets else None}

    def get_result(self):
        return self.client.get_batch_prediction_result(self.batch_prediction_id)

    def get_connector_errors(self):
        return self.client.get_batch_prediction_connector_errors(self.batch_prediction_id)

    def refresh(self):
        self = self.describe()
        return self

    def describe(self):
        return self.client.describe_batch_prediction(self.batch_prediction_id)

    def wait_for_predictions(self, timeout=1200):
        return self.client._poll(self, {'PENDING', 'UPLOADING', 'PREDICTING'}, timeout=timeout)

    def get_status(self):
        return self.describe().status
