import io
import time
from concurrent.futures import ThreadPoolExecutor

from .return_class import AbstractApiClass


class Upload(AbstractApiClass):
    """
        A Upload Reference for uploading file parts
    """

    def __init__(self, client, uploadId=None, datasetUploadId=None, status=None, datasetId=None, datasetVersion=None, modelVersion=None, batchPredictionId=None, parts=None, createdAt=None):
        super().__init__(client, uploadId)
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
        return f"Upload(upload_id={repr(self.upload_id)},\n  dataset_upload_id={repr(self.dataset_upload_id)},\n  status={repr(self.status)},\n  dataset_id={repr(self.dataset_id)},\n  dataset_version={repr(self.dataset_version)},\n  model_version={repr(self.model_version)},\n  batch_prediction_id={repr(self.batch_prediction_id)},\n  parts={repr(self.parts)},\n  created_at={repr(self.created_at)})"

    def to_dict(self):
        return {'upload_id': self.upload_id, 'dataset_upload_id': self.dataset_upload_id, 'status': self.status, 'dataset_id': self.dataset_id, 'dataset_version': self.dataset_version, 'model_version': self.model_version, 'batch_prediction_id': self.batch_prediction_id, 'parts': self.parts, 'created_at': self.created_at}

    def cancel(self):
        """Cancels an upload"""
        return self.client.cancel_upload(self.upload_id)

    def part(self, part_number, part_data):
        """Uploads a part of a large dataset file from your bucket to our system. Our system currently supports a size of up to 5GB for a part of a full file and a size of up to 5TB for the full file. Note that each part must be >=5MB in size, unless it is the last part in the sequence of parts for the full file."""
        return self.client.upload_part(self.upload_id, part_number, part_data)

    def mark_complete(self):
        """Marks an upload process as complete."""
        return self.client.mark_upload_complete(self.upload_id)

    def refresh(self):
        """Calls describe and refreshes the current object's fields"""
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """Retrieves the current upload status (complete or inspecting) and the list of file parts uploaded for a specified dataset upload."""
        return self.client.describe_upload(self.upload_id)

    def upload_part(self, upload_args):
        """
        Uploads a file part. If the upload fails, it will retry up to 3 times with a short backoff before raising an exception.

        Returns:
            UploadPart (json): The object 'UploadPart' that encapsulates the hash and the etag for the part that got uploaded.
        """
        (part_number, part_data) = upload_args
        retries = 0
        while True:
            try:
                return self.part(part_number, part_data)
            except Exception:
                if retries > 2:
                    raise
                part_data.seek(0, 0)
                retries += 1
                time.sleep(retries)

    def upload_file(self, file, threads=10, chunksize=1024 * 1024 * 10, wait_timeout=600):
        """
        Uploads the file in the specified chunk size using the specified number of workers.

        Args:
            file (IOBase): A bytesIO or StringIO object to upload to Abacus.AI
            threads (int, optional): The max number of workers to use while uploading the file
            chunksize (int, optional): The number of bytes to use for each chunk while uploading the file. Defaults to 10 MB
            wait_timeout (int, optional): The max number of seconds to wait for the file parts to be joined on Abacus.AI. Defaults to 600.

        Returns:
            UploadObject(object): The upload file object.
        """
        with ThreadPoolExecutor(max_workers=threads) as pool:
            pool.map(self.upload_part, self._yield_upload_part(file, chunksize))
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
            if not length:
                if part_number == 0:
                    raise Exception('File is empty')
                break
            chunk.seek(0, 0)
            part_number += 1
            yield part_number, chunk

    def wait_for_join(self, timeout=600):
        """
        A waiting call until the upload parts are joined.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to have timed out. Defaults to 600.

        Returns:
            None
        """
        return self.client._poll(self, {'PENDING', 'JOINING'}, timeout=timeout)

    def get_status(self):
        """
        Gets the status of the upload.

        Returns:
            Enum (string): A string describing the status of the upload (pending, complete, etc.).
        """
        return self.describe().status
