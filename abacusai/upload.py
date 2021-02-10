from multiprocessing import Pool
import io
import time


class Upload():
    '''

    '''

    def __init__(self, client, uploadId=None, datasetUploadId=None, status=None, datasetId=None, datasetVersion=None, modelVersion=None, batchPredictionId=None, parts=None, createdAt=None):
        self.client = client
        self.id = uploadId
        self.upload_id = uploadId
        self.dataset_upload_id = datasetUploadId
        self.status = status
        self.dataset_id = datasetId
        self.dataset_version = datasetVersion
        self.model_version = modelVersion
        self.batch_prediction_id = batchPredictionId
        self.parts = parts
        self.created_at = createdAt

    def __repr__(self):
        return f"Upload(upload_id={repr(self.upload_id)}, dataset_upload_id={repr(self.dataset_upload_id)}, status={repr(self.status)}, dataset_id={repr(self.dataset_id)}, dataset_version={repr(self.dataset_version)}, model_version={repr(self.model_version)}, batch_prediction_id={repr(self.batch_prediction_id)}, parts={repr(self.parts)}, created_at={repr(self.created_at)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'upload_id': self.upload_id, 'dataset_upload_id': self.dataset_upload_id, 'status': self.status, 'dataset_id': self.dataset_id, 'dataset_version': self.dataset_version, 'model_version': self.model_version, 'batch_prediction_id': self.batch_prediction_id, 'parts': self.parts, 'created_at': self.created_at}

    def cancel(self):
        return self.client.cancel_upload(self.upload_id)

    def part(self, part_number, part_data):
        return self.client.upload_part(self.upload_id, part_number, part_data)

    def mark_complete(self):
        return self.client.mark_upload_complete(self.upload_id)

    def refresh(self):
        self = self.describe()
        return self

    def describe(self):
        return self.client.describe_upload(self.upload_id)

    def upload_part(self, part_number, part_data):
        '''Uploads a file part.
           If the upload fails, it will retry up to 3 times with a short backoff before raising an exception.
        '''
        retries = 0
        while True:
            try:
                return self.part(part_number, part_data)
            except Exception:
                if retries > 2:
                    raise
                retries += 1
                time.sleep(retries)

    def upload_file(self, file, threads=10, chunksize=1024 * 1024 * 10, wait_timeout=600):
        with Pool(processes=threads) as pool:
            pool.starmap(self.upload_part,
                         self._yield_upload_part(file, chunksize))
        upload_object = self.mark_complete()
        self.wait_for_join(timeout=wait_timeout)
        if upload_object.batch_prediction_id:
            return self.client.describe_batch_prediction(upload_object.batch_prediction_id)
        elif upload_object.dataset_id:
            return self.client.describe_dataset(upload_object.dataset_id)
        return upload_object

    def _yield_upload_part(self, file, chunksize):
        binary_file = isinstance(file, (io.RawIOBase, io.BufferedIOBase))
        part_number = 0
        while True:
            chunk = io.BytesIO() if binary_file else io.StringIO()
            length = chunk.write(file.read(chunksize))
            if length == 0:
                break
            chunk.seek(0, 0)
            part_number += 1
            yield part_number if length else -1, chunk

    def wait_for_join(self, timeout=600):
        return self.client._poll(self, {'PENDING', 'JOINING'}, timeout=timeout)

    def get_status(self):
        return self.describe().status
