import io
import time
from concurrent.futures import ThreadPoolExecutor

from .return_class import AbstractApiClass


class Upload(AbstractApiClass):
    """
        A Upload Reference for uploading file parts

        Args:
            client (ApiClient): An authenticated API Client instance
            uploadId (str): The unique ID generated when the upload process of the full large file in smaller parts is initiated.
            datasetUploadId (str): Same as upload_id. It is kept for backwards compatibility purposes.
            status (str): The current status of the upload.
            datasetId (str): A reference to the dataset this upload is adding data to.
            datasetVersion (str): A reference to the dataset version the upload is adding data to.
            modelId (str): A reference the model the upload is creating a version for
            modelVersion (str): A reference to the model version the upload is creating.
            batchPredictionId (str): A reference to the batch prediction the upload is creating.
            parts (list[dict]): A list containing the order of the file parts that have been uploaded.
            createdAt (str): The timestamp at which the upload was created.
    """

    def __init__(self, client, uploadId=None, datasetUploadId=None, status=None, datasetId=None, datasetVersion=None, modelId=None, modelVersion=None, batchPredictionId=None, parts=None, createdAt=None):
        super().__init__(client, uploadId)
        self.upload_id = uploadId
        self.dataset_upload_id = datasetUploadId
        self.status = status
        self.dataset_id = datasetId
        self.dataset_version = datasetVersion
        self.model_id = modelId
        self.model_version = modelVersion
        self.batch_prediction_id = batchPredictionId
        self.parts = parts
        self.created_at = createdAt

    def __repr__(self):
        return f"Upload(upload_id={repr(self.upload_id)},\n  dataset_upload_id={repr(self.dataset_upload_id)},\n  status={repr(self.status)},\n  dataset_id={repr(self.dataset_id)},\n  dataset_version={repr(self.dataset_version)},\n  model_id={repr(self.model_id)},\n  model_version={repr(self.model_version)},\n  batch_prediction_id={repr(self.batch_prediction_id)},\n  parts={repr(self.parts)},\n  created_at={repr(self.created_at)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'upload_id': self.upload_id, 'dataset_upload_id': self.dataset_upload_id, 'status': self.status, 'dataset_id': self.dataset_id, 'dataset_version': self.dataset_version, 'model_id': self.model_id, 'model_version': self.model_version, 'batch_prediction_id': self.batch_prediction_id, 'parts': self.parts, 'created_at': self.created_at}

    def cancel(self):
        """
        Cancels an upload.

        Args:
            upload_id (str): A unique string identifier for the upload.
        """
        return self.client.cancel_upload(self.upload_id)

    def part(self, part_number: int, part_data: io.TextIOBase):
        """
        Uploads part of a large dataset file from your bucket to our system. Our system currently supports parts of up to 5GB and full files of up to 5TB. Note that each part must be at least 5MB in size, unless it is the last part in the sequence of parts for the full file.

        Args:
            part_number (int): The 1-indexed number denoting the position of the file part in the sequence of parts for the full file.
            part_data (io.TextIOBase): The multipart/form-data for the current part of the full file.

        Returns:
            UploadPart: The object 'UploadPart' which encapsulates the hash and the etag for the part that got uploaded.
        """
        return self.client.upload_part(self.upload_id, part_number, part_data)

    def mark_complete(self):
        """
        Marks an upload process as complete.

        Args:
            upload_id (str): A unique string identifier for the upload process.

        Returns:
            Upload: The upload object associated with the process, containing details of the file.
        """
        return self.client.mark_upload_complete(self.upload_id)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            Upload: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Retrieves the current upload status (complete or inspecting) and the list of file parts uploaded for a specified dataset upload.

        Args:
            upload_id (str): The unique ID associated with the file uploaded or being uploaded in parts.

        Returns:
            Upload: Details associated with the large dataset file uploaded in parts.
        """
        return self.client.describe_upload(self.upload_id)

    def upload_part(self, upload_args):
        """
        Uploads a file part. If the upload fails, it will retry up to 3 times with a short backoff before raising an exception.

        Returns:
            UploadPart: The object 'UploadPart' that encapsulates the hash and the etag for the part that got uploaded.
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
            Upload: The upload file object.
        """
        with ThreadPoolExecutor(max_workers=threads) as pool:
            pool.map(self.upload_part, self._yield_upload_part(file, chunksize))
        upload_object = self.mark_complete()
        self.wait_for_join(timeout=wait_timeout)
        if upload_object.batch_prediction_id:
            return self.client.describe_batch_prediction(upload_object.batch_prediction_id)
        elif upload_object.dataset_id:
            return self.client.describe_dataset(upload_object.dataset_id)
        elif upload_object.model_id:
            return self.client.describe_model(upload_object.model_id)
        return upload_object

    def _yield_upload_part(self, file, chunksize):
        part_number = 0
        while True:
            chunk_data = file.read(chunksize)
            if not chunk_data:
                if part_number == 0:
                    raise Exception('File is empty')
                break
            chunk = io.StringIO(chunk_data) if isinstance(
                chunk_data, str) else io.BytesIO(chunk_data)
            part_number += 1
            yield part_number, chunk

    def wait_for_join(self, timeout=600):
        """
        A waiting call until the upload parts are joined.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to have timed out. Defaults to 600.
        """
        return self.client._poll(self, {'PENDING', 'JOINING'}, timeout=timeout)

    def get_status(self):
        """
        Gets the status of the upload.

        Returns:
            str: A string describing the status of the upload (pending, complete, etc.).
        """
        return self.describe().status
