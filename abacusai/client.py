import inspect
import io
import logging
import time
from typing import List

import requests
from packaging import version

from .api_key import ApiKey
from .batch_prediction import BatchPrediction
from .batch_prediction_version import BatchPredictionVersion
from .schema import Schema
from .data_filter import DataFilter
from .database_connector import DatabaseConnector
from .dataset import Dataset
from .dataset_version import DatasetVersion
from .deployment import Deployment
from .deployment_auth_token import DeploymentAuthToken
from .feature import Feature
from .feature_column import FeatureColumn
from .feature_group import FeatureGroup
from .feature_group_export import FeatureGroupExport
from .feature_record import FeatureRecord
from .file_connector import FileConnector
from .file_connector_instructions import FileConnectorInstructions
from .file_connector_verification import FileConnectorVerification
from .model import Model
from .model_location import ModelLocation
from .model_metrics import ModelMetrics
from .model_upload import ModelUpload
from .model_version import ModelVersion
from .prediction_dataset import PredictionDataset
from .prediction_feature_group import PredictionFeatureGroup
from .prediction_input import PredictionInput
from .project import Project
from .project_dataset import ProjectDataset
from .project_validation import ProjectValidation
from .refresh_pipeline_run import RefreshPipelineRun
from .refresh_policy import RefreshPolicy
from .streaming_auth_token import StreamingAuthToken
from .training_config_options import TrainingConfigOptions
from .upload import Upload
from .upload_part import UploadPart
from .use_case import UseCase
from .use_case_requirements import UseCaseRequirements


class ApiException(Exception):
    def __init__(self, message, http_status, exception=None):
        self.message = message
        self.http_status = http_status
        self.exception = exception or 'ApiException'

    def __str__(self):
        return f'{self.exception}({self.http_status}): {self.message}'


class ApiClient():
    client_version = '0.18.0'

    def __init__(self, api_key=None, server='https://abacus.ai'):
        self.api_key = api_key
        self.web_version = None
        self.server = server
        self.user = None
        # Connection and version check
        try:
            self.web_version = self._call_api(
                'version', 'GET', server_override='https://abacus.ai')
            if version.parse(self.web_version) > version.parse(self.client_version):
                logging.warning(
                    'A new version of the Abacus.AI library is available')
                logging.warning(
                    f'Current Version: {self.client_version} -> New Version: {self.web_version}')
        except Exception:
            logging.error(
                'Failed get the current API version from Abacus.AI (https://abacus.ai/api/v0/version)')
        if api_key is not None:
            try:
                self.user = self._call_api('getUser', 'GET')
            except Exception:
                logging.error('Invalid API Key')

    def _call_api(
            self, action, method, query_params=None,
            body=None, files=None, parse_type=None, streamable_response=False, server_override=None):
        headers = {'apiKey': self.api_key, 'clientVersion': self.client_version,
                   'User-Agent': f'python-abacusai-{self.client_version}'}
        url = (server_override or self.server) + '/api/v0/' + action

        response = self._request(url, method, query_params=query_params,
                                 headers=headers, body=body, files=files, stream=streamable_response)
        result = None
        success = False
        error_message = None
        error_type = None
        if streamable_response and response.status_code == 200:
            return response.raw
        try:
            json_data = response.json()
            success = json_data['success']
            error_message = json_data.get('error')
            error_type = json_data.get('errorType')
            result = json_data.get('result')
            if success and parse_type:
                result = self._build_class(parse_type, result)
        except Exception:
            error_message = response.text
        if not success:
            if response.status_code > 502 and response.status_code not in (501, 503):
                error_message = 'Internal Server Error, please contact dev@abacus.ai for support'
            raise ApiException(error_message, response.status_code, error_type)
        return result

    def _build_class(self, return_class, values):
        if values is None:
            return None
        if isinstance(values, list):
            return [self._build_class(return_class, val) for val in values if val is not None]
        type_inputs = inspect.signature(return_class.__init__).parameters
        return return_class(self, **{key: val for key, val in values.items() if key in type_inputs})

    def _request(self, url, method, query_params=None, headers=None,
                 body=None, files=None, stream=False):
        if method == 'GET':
            return requests.get(url, params=query_params, headers=headers, stream=stream)
        elif method == 'POST':
            return requests.post(url, params=query_params, json=body, headers=headers, files=files, timeout=90)
        elif method == 'PUT':
            return requests.put(url, params=query_params, data=body, headers=headers, files=files, timeout=90)
        elif method == 'PATCH':
            return requests.patch(url, params=query_params, json=body, headers=headers, files=files, timeout=90)
        elif method == 'DELETE':
            return requests.delete(url, params=query_params, data=body, headers=headers)
        else:
            raise ValueError(
                'HTTP method must be `GET`, `POST`, `PATCH`, `PUT` or `DELETE`'
            )

    def _poll(self, obj, wait_states: set, delay: int = 5, timeout: int = 300, poll_args: dict = {}):
        start_time = time.time()
        while obj.get_status(**poll_args) in wait_states:
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f'Maximum wait time of {timeout}s exceeded')
            time.sleep(delay)
        return obj.describe()

    def add_user_to_organization(self, email: str):
        '''Invites a user to your organization. This method will send the specified email address an invitation link to join your organization.'''
        return self._call_api('addUserToOrganization', 'POST', query_params={}, body={'email': email})

    def list_api_keys(self):
        '''Lists all of the API keys created by the current member in the user's organization.'''
        return self._call_api('listApiKeys', 'GET', query_params={}, parse_type=ApiKey)

    def list_organization_users(self):
        '''Retrieves a list of all users in the organization.

        This method will retrieve a list containing all the users in the organization and their account information. The list includes both, users who have accepted their invitation and users who have not yet accepted their invitation.
        '''
        return self._call_api('listOrganizationUsers', 'GET', query_params={})

    def get_user(self):
        '''Get the current user's information, such as their name, email, admin status, etc.'''
        return self._call_api('getUser', 'GET', query_params={})

    def set_user_as_admin(self, email: str):
        '''Sets the specified user as an Organization Administrator. You must be an Organization Administrator to use this method. An Organization Administrator manages billing and the organization's users.'''
        return self._call_api('setUserAsAdmin', 'POST', query_params={}, body={'email': email})

    def delete_api_key(self, api_key_id: str):
        '''Delete a specified API Key. You can use the "listApiKeys" method to find the list of all API Key IDs.'''
        return self._call_api('deleteApiKey', 'DELETE', query_params={'apiKeyId': api_key_id})

    def remove_user_from_organization(self, email: str):
        '''Removes the specified user from the organization. You can remove yourself or you must be an Organization Administrator to use this method to remove other users from the organization.'''
        return self._call_api('removeUserFromOrganization', 'DELETE', query_params={'email': email})

    def create_project(self, name: str, use_case: str):
        '''Creates a project with your specified project name and use case. Creating a project creates a container for all of the datasets and the models that are associated with a particular problem/project that you would like to work on. For example, if you want to create a model to detect fraud, you have to first create a project, upload the associated datasets, and then create one or more models to get predictions for your use case.'''
        return self._call_api('createProject', 'GET', query_params={'name': name, 'useCase': use_case}, parse_type=Project)

    def list_use_cases(self):
        '''Retrieves a list of all use cases with descriptions. Use the given mappings to specify a use case when needed.'''
        return self._call_api('listUseCases', 'GET', query_params={}, parse_type=UseCase)

    def describe_use_case_requirements(self, use_case: str):
        '''This API call returns the dataset requirements for a specified use case. Use this API if you're unsure about what your dataset needs.'''
        return self._call_api('describeUseCaseRequirements', 'GET', query_params={'useCase': use_case}, parse_type=UseCaseRequirements)

    def describe_project(self, project_id: str):
        '''Returns a description of a project. The method takes the ID of the project as an input.'''
        return self._call_api('describeProject', 'GET', query_params={'projectId': project_id}, parse_type=Project)

    def list_projects(self):
        '''Retrieves a list of all projects in the current organization.'''
        return self._call_api('listProjects', 'GET', query_params={}, parse_type=Project)

    def list_project_datasets(self, project_id: str):
        '''Retrieves all dataset(s) attached to a specified project. This API returns all attributes of each dataset, such as its name, type, and ID.'''
        return self._call_api('listProjectDatasets', 'GET', query_params={'projectId': project_id}, parse_type=ProjectDataset)

    def get_schema(self, project_id: str, dataset_id: str):
        '''Returns a schema given a specific dataset in a project. The schema of the dataset consists of the columns in the dataset, the data type of the column, and the column's column mapping.'''
        return self._call_api('getSchema', 'GET', query_params={'projectId': project_id, 'datasetId': dataset_id}, parse_type=Schema)

    def rename_project(self, project_id: str, name: str):
        '''This method renames a project after it is created. To rename the project, specify its ID and new name.'''
        return self._call_api('renameProject', 'PATCH', query_params={}, body={'projectId': project_id, 'name': name})

    def set_column_data_type(self, project_id: str, dataset_id: str, column: str, data_type: str):
        '''Set a column's type in a specified dataset. Specify the project ID, dataset ID, column name and data type, and the method will return the entire dataset with the resulting changes reflected.'''
        return self._call_api('setColumnDataType', 'POST', query_params={'datasetId': dataset_id}, body={'projectId': project_id, 'column': column, 'dataType': data_type}, parse_type=Schema)

    def set_column_mapping(self, project_id: str, dataset_id: str, column: str, column_mapping: str):
        '''Set a column's column mapping. If the column mapping is single-use and already set in another column in this dataset, this call will first remove the other column's mapping and move it to this column. The model returns a list of all schemas for each column with the reflected changes.'''
        return self._call_api('setColumnMapping', 'POST', query_params={'datasetId': dataset_id}, body={'projectId': project_id, 'column': column, 'columnMapping': column_mapping}, parse_type=Schema)

    def add_custom_column(self, project_id: str, dataset_id: str, column: str, select_expression: str):
        '''Adds a custom column to the dataset. To add a column, the user needs to provide the location of the dataset to add a custom column to. This is performed by providing a project ID and a dataset ID. The method also requires a custom column name and a SELECT SQL statement to generate the new column.'''
        return self._call_api('addCustomColumn', 'POST', query_params={'datasetId': dataset_id}, body={'projectId': project_id, 'column': column, 'selectExpression': select_expression}, parse_type=Schema)

    def edit_custom_column(self, project_id: str, dataset_id: str, column: str, new_column_name: str = None, select_expression: str = None):
        '''Edits a custom column'''
        return self._call_api('editCustomColumn', 'PATCH', query_params={'datasetId': dataset_id}, body={'projectId': project_id, 'column': column, 'newColumnName': new_column_name, 'selectExpression': select_expression}, parse_type=Schema)

    def delete_custom_column(self, project_id: str, dataset_id: str, column: str):
        '''Deletes a custom column'''
        return self._call_api('deleteCustomColumn', 'DELETE', query_params={'projectId': project_id, 'datasetId': dataset_id, 'column': column}, parse_type=Schema)

    def set_project_dataset_filters(self, project_id: str, dataset_id: str, filters: list):
        '''Sets the data filters for a dataset uploaded under a project.

        Each filter in the filter list must be an object containing the keys 'type' and 'whereExpression'. The type must be either 'INCLUDE', signifying any rows matching the sql statement will be included as part of training, or 'EXCLUDE' signifying that any rows matching the sql statement will be excluded from training.
        '''
        return self._call_api('setProjectDatasetFilters', 'POST', query_params={'datasetId': dataset_id}, body={'projectId': project_id, 'filters': filters})

    def validate_project(self, project_id: str):
        '''Validates that the specified project has all required datasets for its use case and that all datasets attached to the project have all required columns.'''
        return self._call_api('validateProject', 'GET', query_params={'projectId': project_id}, parse_type=ProjectValidation)

    def remove_column_mapping(self, project_id: str, dataset_id: str, column: str):
        '''Removes a column mapping from a column in the dataset. The model will display a list of all columns with their mappings once the change is made.'''
        return self._call_api('removeColumnMapping', 'DELETE', query_params={'projectId': project_id, 'datasetId': dataset_id, 'column': column}, parse_type=Schema)

    def delete_project(self, project_id: str):
        '''Deletes a specified project from your organization.

        This method deletes the project, trained models and deployments in the specified project. The datasets attached to the specified project remain available for use with other projects in the organization.

        This method will not delete a project that contains active deployments. Be sure to stop all active deployments before you use the delete option.

        Note: All projects, models, and deployments cannot be recovered once they are deleted.
        '''
        return self._call_api('deleteProject', 'DELETE', query_params={'projectId': project_id})

    def cancel_upload(self, upload_id: str):
        '''Cancels an upload'''
        return self._call_api('cancelUpload', 'DELETE', query_params={'uploadId': upload_id})

    def upload_part(self, upload_id: str, part_number: int, part_data: io.TextIOBase):
        '''Uploads a part of a large dataset file from your bucket to our system. Our system currently supports a size of up to 5GB for a part of a full file and a size of up to 5TB for the full file. Note that each part must be >=5MB in size, unless it is the last part in the sequence of parts for the full file.'''
        return self._call_api('uploadPart', 'POST', query_params={'uploadId': upload_id, 'partNumber': part_number}, parse_type=UploadPart, files={'partData': part_data})

    def mark_upload_complete(self, upload_id: str):
        '''Marks an upload process as complete. This can be used to signify that the upload of the full file on our system is complete as all the parts/chunks associated with the full file are successfully uploaded.'''
        return self._call_api('markUploadComplete', 'POST', query_params={}, body={'uploadId': upload_id}, parse_type=Upload)

    def create_dataset_from_file_connector(self, name: str, location: str, file_format: str = None, project_id: str = None, dataset_type: str = None, refresh_schedule: str = None):
        '''Creates a dataset from a file located in a cloud storage, such as Amazon AWS S3, using the specified dataset name and location. The model will return the dataset's information, such as its ID, name, data source, etc.'''
        return self._call_api('createDatasetFromFileConnector', 'POST', query_params={}, body={'name': name, 'location': location, 'fileFormat': file_format, 'projectId': project_id, 'datasetType': dataset_type, 'refreshSchedule': refresh_schedule}, parse_type=Dataset)

    def create_dataset_version_from_file_connector(self, dataset_id: str, location: str = None, file_format: str = None):
        '''Creates a new version of the specified dataset. The model returns the new version of the dataset with its attributes.'''
        return self._call_api('createDatasetVersionFromFileConnector', 'POST', query_params={'datasetId': dataset_id}, body={'location': location, 'fileFormat': file_format}, parse_type=DatasetVersion)

    def create_dataset_from_database_connector(self, name: str, database_connector_id: str, object_name: str = None, columns: str = None, query_arguments: str = None, project_id: str = None, dataset_type: str = None, refresh_schedule: str = None):
        '''Creates a dataset from a Database Connector'''
        return self._call_api('createDatasetFromDatabaseConnector', 'POST', query_params={}, body={'name': name, 'databaseConnectorId': database_connector_id, 'objectName': object_name, 'columns': columns, 'queryArguments': query_arguments, 'projectId': project_id, 'datasetType': dataset_type, 'refreshSchedule': refresh_schedule}, parse_type=Dataset)

    def create_dataset_version_from_database_connector(self, dataset_id: str, object_name: str = None, columns: str = None, query_arguments: str = None):
        '''Creates a new version of the specified dataset'''
        return self._call_api('createDatasetVersionFromDatabaseConnector', 'POST', query_params={'datasetId': dataset_id}, body={'objectName': object_name, 'columns': columns, 'queryArguments': query_arguments}, parse_type=DatasetVersion)

    def create_dataset_from_upload(self, name: str, file_format: str = None, project_id: str = None, dataset_type: str = None):
        '''Creates a dataset and return an upload Id that can be used to upload a file. The model will take in the name of your file and return the dataset's information (its attributes).'''
        return self._call_api('createDatasetFromUpload', 'POST', query_params={}, body={'name': name, 'fileFormat': file_format, 'projectId': project_id, 'datasetType': dataset_type}, parse_type=Upload)

    def create_dataset_version_from_upload(self, dataset_id: str, file_format: str = None):
        '''Creates a new version of the specified dataset using a local file upload.'''
        return self._call_api('createDatasetVersionFromUpload', 'POST', query_params={'datasetId': dataset_id}, body={'fileFormat': file_format}, parse_type=Upload)

    def create_streaming_dataset(self, name: str, project_id: str, dataset_type: str):
        '''Creates a streaming dataset. Use a streaming dataset if your dataset is receiving information from multiple sources over an extended period of time.'''
        return self._call_api('createStreamingDataset', 'POST', query_params={}, body={'name': name, 'projectId': project_id, 'datasetType': dataset_type}, parse_type=Dataset)

    def snapshot_streaming_data(self, dataset_id: str):
        '''Snapshots the current data in the streaming dataset for training.'''
        return self._call_api('snapshotStreamingData', 'POST', query_params={'datasetId': dataset_id}, body={}, parse_type=DatasetVersion)

    def create_dataset(self, name: str, location: str, file_format: str = None, project_id: str = None, dataset_type: str = None, refresh_schedule: str = None):
        '''[DEPRECATED] Creates a dataset from a file located in a cloud storage, such as Amazon AWS S3, using the specified dataset name and location. The model will return the dataset's information, such as its ID, name, data source, etc.'''
        logging.warning(
            'This function is deprecated and will be removed in a future version. Use create_dataset_from_file_connector instead.')
        return self._call_api('createDataset', 'POST', query_params={}, body={'name': name, 'location': location, 'fileFormat': file_format, 'projectId': project_id, 'datasetType': dataset_type, 'refreshSchedule': refresh_schedule}, parse_type=Dataset)

    def create_dataset_version(self, dataset_id: str, location: str = None, file_format: str = None):
        '''[DEPRECATED] Creates a new version of the specified dataset. The model returns the new version of the dataset with its attributes.'''
        logging.warning(
            'This function is deprecated and will be removed in a future version. Use create_dataset_version_from_file_connector instead.')
        return self._call_api('createDatasetVersion', 'POST', query_params={'datasetId': dataset_id}, body={'location': location, 'fileFormat': file_format}, parse_type=DatasetVersion)

    def create_dataset_from_local_file(self, name: str, file_format: str = None, project_id: str = None, dataset_type: str = None):
        '''[DEPRECATED] Creates a dataset and return an upload Id that can be used to upload a file. The model will take in the name of your file and return the dataset's information (its attributes).'''
        logging.warning(
            'This function is deprecated and will be removed in a future version. Use create_dataset_from_upload instead.')
        return self._call_api('createDatasetFromLocalFile', 'POST', query_params={}, body={'name': name, 'fileFormat': file_format, 'projectId': project_id, 'datasetType': dataset_type}, parse_type=Upload)

    def create_dataset_version_from_local_file(self, dataset_id: str, file_format: str = None):
        '''[DEPRECATED] Creates a new version of the specified dataset using a local file upload.'''
        logging.warning(
            'This function is deprecated and will be removed in a future version. Use create_dataset_version_from_upload instead.')
        return self._call_api('createDatasetVersionFromLocalFile', 'POST', query_params={'datasetId': dataset_id}, body={'fileFormat': file_format}, parse_type=Upload)

    def get_file_connector_instructions(self, bucket: str, write_permission: bool = False):
        '''Retrieves verification information to create a data connector to a cloud storage bucket.'''
        return self._call_api('getFileConnectorInstructions', 'GET', query_params={'bucket': bucket, 'writePermission': write_permission}, parse_type=FileConnectorInstructions)

    def list_database_connectors(self):
        '''Retrieves a list of all of the database connectors along with all their attributes.'''
        return self._call_api('listDatabaseConnectors', 'GET', query_params={}, parse_type=DatabaseConnector)

    def list_file_connectors(self):
        '''Retrieves a list of all connected services in the organization and their current verification status.'''
        return self._call_api('listFileConnectors', 'GET', query_params={}, parse_type=FileConnector)

    def list_database_connector_objects(self, database_connector_id: str):
        '''Lists querable objects in the database connector.'''
        return self._call_api('listDatabaseConnectorObjects', 'GET', query_params={'databaseConnectorId': database_connector_id})

    def get_database_connector_object_schema(self, database_connector_id: str, object_name: str = None):
        '''Get the schema of an object in an database connector.'''
        return self._call_api('getDatabaseConnectorObjectSchema', 'GET', query_params={'databaseConnectorId': database_connector_id, 'objectName': object_name})

    def rename_database_connector(self, database_connector_id: str, name: str):
        '''Renames a Database Connector'''
        return self._call_api('renameDatabaseConnector', 'PATCH', query_params={}, body={'databaseConnectorId': database_connector_id, 'name': name})

    def verify_database_connector(self, database_connector_id: str):
        '''Checks to see if Abacus.AI can access the database.'''
        return self._call_api('verifyDatabaseConnector', 'GET', query_params={'databaseConnectorId': database_connector_id})

    def verify_file_connector(self, bucket: str):
        '''Checks to see if Abacus.AI can access the bucket.'''
        return self._call_api('verifyFileConnector', 'POST', query_params={}, body={'bucket': bucket}, parse_type=FileConnectorVerification)

    def remove_database_connector(self, database_connector_id: str):
        '''Disconnect an database connector.'''
        return self._call_api('removeDatabaseConnector', 'DELETE', query_params={'databaseConnectorId': database_connector_id})

    def remove_file_connector(self, bucket: str):
        '''Removes a connected service from the specified organization.'''
        return self._call_api('removeFileConnector', 'DELETE', query_params={'bucket': bucket})

    def create_streaming_token(self):
        '''Creates a streaming token for the specified project. Streaming tokens are used to authenticate requests to append data to streaming datasets.'''
        return self._call_api('createStreamingToken', 'POST', query_params={}, body={}, parse_type=StreamingAuthToken)

    def list_streaming_tokens(self):
        '''Retrieves a list of all streaming tokens along with their attributes.'''
        return self._call_api('listStreamingTokens', 'GET', query_params={}, parse_type=StreamingAuthToken)

    def delete_streaming_token(self, streaming_token: str):
        '''Deletes the specified streaming token. Note that the streaming tokens are not recoverable once they are deleted.'''
        return self._call_api('deleteStreamingToken', 'DELETE', query_params={'streamingToken': streaming_token})

    def list_uploads(self):
        '''Lists all the details associated with the dataset file upload including the current status of the upload (complete or not). Use this API method to ensure that your dataset is uploaded properly and all the details associated with it are correct. If everything looks fine then you are good to use the dataset you uploaded in parts for your batch predictins and/or model training.'''
        return self._call_api('listUploads', 'GET', query_params={}, parse_type=Upload)

    def describe_upload(self, upload_id: str):
        '''Retrieves the current upload status (complete or inspecting) and the list of file parts uploaded for a specified dataset upload.'''
        return self._call_api('describeUpload', 'GET', query_params={'uploadId': upload_id}, parse_type=Upload)

    def list_datasets(self):
        '''Retrieves a list of all of the datasets in the organization, each with their attributes and IDs.'''
        return self._call_api('listDatasets', 'GET', query_params={}, parse_type=Dataset)

    def describe_dataset(self, dataset_id: str):
        '''Retrieves a full description of the specified dataset, with attributes such as its ID, name, source type, etc.'''
        return self._call_api('describeDataset', 'GET', query_params={'datasetId': dataset_id}, parse_type=Dataset)

    def list_dataset_versions(self, dataset_id: str):
        '''Retrieves a list of all dataset versions for the specified dataset.'''
        return self._call_api('listDatasetVersions', 'GET', query_params={'datasetId': dataset_id}, parse_type=DatasetVersion)

    def attach_dataset_to_project(self, dataset_id: str, project_id: str, dataset_type: str):
        '''Attaches the dataset to the project.

        Use this method to attach a dataset that is already in the organization to another project. The dataset type is required to let the AI engine know what type of schema should be used.
        '''
        return self._call_api('attachDatasetToProject', 'POST', query_params={'datasetId': dataset_id}, body={'projectId': project_id, 'datasetType': dataset_type}, parse_type=Schema)

    def remove_dataset_from_project(self, dataset_id: str, project_id: str):
        '''Removes a dataset from a project.'''
        return self._call_api('removeDatasetFromProject', 'POST', query_params={'datasetId': dataset_id}, body={'projectId': project_id})

    def rename_dataset(self, dataset_id: str, name: str):
        '''Rename a dataset that has already been defined. Specify the new name and dataset ID, and the model will return the attributes of the renamed dataset.'''
        return self._call_api('renameDataset', 'PATCH', query_params={'datasetId': dataset_id}, body={'name': name})

    def delete_dataset(self, dataset_id: str):
        '''Deletes the specified dataset from the organization.

        The dataset cannot be deleted if it is currently attached to a project.
        '''
        return self._call_api('deleteDataset', 'DELETE', query_params={'datasetId': dataset_id})

    def get_training_config_options(self, project_id: str):
        '''Retrieves the full description of the model training configuration options available for the specified project.

        The number of configuration options and types of options available is determined by the use case associated with the specified project. Refer to the (Use Case Documentation)[{USE_CASE_URL}] for more information on use cases and use case specific configuration options.
        '''
        return self._call_api('getTrainingConfigOptions', 'GET', query_params={'projectId': project_id}, parse_type=TrainingConfigOptions)

    def train_model(self, project_id: str, name: str = None, training_config: dict = {}, refresh_schedule: str = None):
        '''Trains a model for the specified project.

        Use this method to train a model with default training configurations for the specified project. This method also supports user-specified training configurations available by using the getTrainingConfigOptions method.
        '''
        return self._call_api('trainModel', 'POST', query_params={}, body={'projectId': project_id, 'name': name, 'trainingConfig': training_config, 'refreshSchedule': refresh_schedule}, parse_type=Model)

    def list_models(self, project_id: str):
        '''Retrieves the list of models in the specified project.'''
        return self._call_api('listModels', 'GET', query_params={'projectId': project_id}, parse_type=Model)

    def describe_model(self, model_id: str):
        '''Retrieves a full description of the specified model.'''
        return self._call_api('describeModel', 'GET', query_params={'modelId': model_id}, parse_type=Model)

    def update_model_training_config(self, model_id: str, training_config: dict):
        '''Edits the default model training config'''
        return self._call_api('updateModelTrainingConfig', 'PATCH', query_params={}, body={'modelId': model_id, 'trainingConfig': training_config}, parse_type=Model)

    def get_model_metrics(self, model_id: str, model_version: str = None, baseline_metrics: bool = False):
        '''Retrieves a full list of the metrics for the specified model.

        If only the model's unique identifier (modelId) is specified, the latest trained version of model (modelVersion) is used.
        '''
        return self._call_api('getModelMetrics', 'GET', query_params={'modelId': model_id, 'modelVersion': model_version, 'baselineMetrics': baseline_metrics}, parse_type=ModelMetrics)

    def list_model_versions(self, model_id: str):
        '''Retrieves a list of all model version for a given model.'''
        return self._call_api('listModelVersions', 'GET', query_params={'modelId': model_id}, parse_type=ModelVersion)

    def retrain_model(self, model_id: str, deployment_ids: list = []):
        '''Retrains the specified model. Gives you an option to choose the deployments you want the retraining to be deployed to.'''
        return self._call_api('retrainModel', 'POST', query_params={}, body={'modelId': model_id, 'deploymentIds': deployment_ids}, parse_type=Model)

    def cancel_model_training(self, model_id: str):
        '''Cancels the training process for the specified model.

        Use this to stop model training.The model status will be set to CANCELLED.
        '''
        return self._call_api('cancelModelTraining', 'DELETE', query_params={'modelId': model_id})

    def delete_model(self, model_id: str):
        '''Deletes the specified model and all its versions. Note that models are not recoverable after they are deleted.'''
        return self._call_api('deleteModel', 'DELETE', query_params={'modelId': model_id})

    def delete_model_version(self, model_version: str):
        '''Deletes the specified model version. Note that models versions are not recoverable after they are deleted.'''
        return self._call_api('deleteModelVersion', 'DELETE', query_params={'modelVersion': model_version})

    def create_deployment(self, model_id: str, name: str = None, description: str = None, calls_per_second: int = None, auto_deploy: bool = True):
        '''Creates a deployment with the specified name and description for the specified model.

        A Deployment makes the trained model available for prediction requests.
        '''
        return self._call_api('createDeployment', 'POST', query_params={}, body={'modelId': model_id, 'name': name, 'description': description, 'callsPerSecond': calls_per_second, 'autoDeploy': auto_deploy}, parse_type=Deployment)

    def create_deployment_token(self, project_id: str):
        '''Creates a deployment token for the specified project.

        Deployment tokens are used to authenticate requests to the prediction APIs and are scoped on the project level.
        '''
        return self._call_api('createDeploymentToken', 'POST', query_params={}, body={'projectId': project_id}, parse_type=DeploymentAuthToken)

    def describe_deployment(self, deployment_id: str):
        '''Retrieves a full description of the specified deployment.'''
        return self._call_api('describeDeployment', 'GET', query_params={'deploymentId': deployment_id}, parse_type=Deployment)

    def list_deployments(self, project_id: str):
        '''Retrieves a list of all deployments in the specified project.'''
        return self._call_api('listDeployments', 'GET', query_params={'projectId': project_id}, parse_type=Deployment)

    def list_deployment_tokens(self, project_id: str):
        '''Retrieves a list of all deployment tokens in the specified project.'''
        return self._call_api('listDeploymentTokens', 'GET', query_params={'projectId': project_id}, parse_type=DeploymentAuthToken)

    def update_deployment(self, deployment_id: str, name: str = None, description: str = None):
        '''Updates a deployment's name and/or description.'''
        return self._call_api('updateDeployment', 'PATCH', query_params={'deploymentId': deployment_id}, body={'name': name, 'description': description})

    def set_auto_deployment(self, deployment_id: str, enable: bool = None):
        '''Enable/Disable auto deployment for the specified deployment.

        When a model is scheduled to retrain, deployments with this enabled will be marked to automatically promote the new model
        version. After the newly trained model completes, a check on its metrics in comparison to the currently deployed model version
        will be performed. If the metrics are comparable or better, the newly trained model version is automatically promoted. If not,
        it will be marked as a failed model version promotion with an error indicating poor metrics performance.
        '''
        return self._call_api('setAutoDeployment', 'POST', query_params={'deploymentId': deployment_id}, body={'enable': enable})

    def set_deployment_model_version(self, deployment_id: str, model_version: str):
        '''Promotes a Model Version to be served in the Deployment'''
        return self._call_api('setDeploymentModelVersion', 'PATCH', query_params={'deploymentId': deployment_id}, body={'modelVersion': model_version})

    def start_deployment(self, deployment_id: str):
        '''Restarts the specified deployment that was previously suspended.'''
        return self._call_api('startDeployment', 'GET', query_params={'deploymentId': deployment_id})

    def stop_deployment(self, deployment_id: str):
        '''Stops the specified deployment.'''
        return self._call_api('stopDeployment', 'GET', query_params={'deploymentId': deployment_id})

    def delete_deployment(self, deployment_id: str):
        '''Deletes the specified deployment. The deployment's models will not be affected. Note that the deployments are not recoverable after they are deleted.'''
        return self._call_api('deleteDeployment', 'DELETE', query_params={'deploymentId': deployment_id})

    def delete_deployment_token(self, deployment_token: str):
        '''Deletes the specified deployment token. Note that the deployment tokens are not recoverable after they are deleted'''
        return self._call_api('deleteDeploymentToken', 'DELETE', query_params={'deploymentToken': deployment_token})

    def predict(self, deployment_token: str, deployment_id: str, query_data: dict = {}):
        '''Returns a prediction for Predictive Modeling'''
        return self._call_api('predict', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data})

    def predict_lead(self, deployment_token: str, deployment_id: str, query_data: dict):
        '''Returns the probability of a user to be a lead on the basis of his/her interaction with the service/product and user's own attributes (e.g. income, assets, credit score, etc.). Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'user_id' mapped to mapping 'LEAD_ID' in our system).'''
        return self._call_api('predictLead', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data})

    def predict_churn(self, deployment_token: str, deployment_id: str, query_data: dict):
        '''Returns a probability of a user to churn out in response to his/her interactions with the item/product/service. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'churn_result' mapped to mapping 'CHURNED_YN' in our system).'''
        return self._call_api('predictChurn', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data})

    def predict_takeover(self, deployment_token: str, deployment_id: str, query_data: dict):
        '''Returns a probability for each class label associated with the types of fraud or a 'yes' or 'no' type label for the possibility of fraud. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'account_name' mapped to mapping 'ACCOUNT_ID' in our system).'''
        return self._call_api('predictTakeover', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data})

    def predict_fraud(self, deployment_token: str, deployment_id: str, query_data: dict):
        '''Returns a probability of a transaction performed under a specific account as being a fraud or not. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'account_number' mapped to the mapping 'ACCOUNT_ID' in our system).'''
        return self._call_api('predictFraud', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data})

    def predict_class(self, deployment_token: str, deployment_id: str, query_data: dict = {}, threshold: float = None, threshold_class: str = None):
        '''Returns a prediction for regression classification'''
        return self._call_api('predictClass', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'threshold': threshold, 'thresholdClass': threshold_class})

    def get_anomalies(self, deployment_token: str, deployment_id: str, threshold: float = None, histogram: bool = False):
        '''Returns a list of anomalies from the training dataset'''
        return self._call_api('getAnomalies', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'threshold': threshold, 'histogram': histogram})

    def is_anomaly(self, deployment_token: str, deployment_id: str, query_data: dict = None):
        '''Returns a list of anomaly attributes based on login information for a specified account. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'account_name' mapped to mapping 'ACCOUNT_ID' in our system).'''
        return self._call_api('isAnomaly', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data})

    def get_forecast(self, deployment_token: str, deployment_id: str, query_data: dict, future_data: dict = None, num_predictions: int = None, prediction_start: str = None):
        '''Returns a list of forecasts for a given entity under the specified project deployment. Note that the inputs to the deployed model will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'holiday_yn' mapped to mapping 'FUTURE' in our system).'''
        return self._call_api('getForecast', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'futureData': future_data, 'numPredictions': num_predictions, 'predictionStart': prediction_start})

    def get_labels(self, deployment_token: str, deployment_id: str, query_data: dict, threshold: float = 0.5):
        '''Returns a list of scored labels from'''
        return self._call_api('getLabels', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'threshold': threshold})

    def get_recommendations(self, deployment_token: str, deployment_id: str, query_data: dict, num_items: int = 50, page: int = 1, exclude_item_ids: list = [], score_field: str = '', scaling_factors: list = [], restrict_items: list = [], exclude_items: list = [], explore_fraction: float = 0.0):
        '''Returns a list of recommendations for a given user under the specified project deployment. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'time' mapped to mapping 'TIMESTAMP' in our system).'''
        return self._call_api('getRecommendations', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'numItems': num_items, 'page': page, 'excludeItemIds': exclude_item_ids, 'scoreField': score_field, 'scalingFactors': scaling_factors, 'restrictItems': restrict_items, 'excludeItems': exclude_items, 'exploreFraction': explore_fraction})

    def get_personalized_ranking(self, deployment_token: str, deployment_id: str, query_data: dict, preserve_ranks: list = [], scaling_factors: list = []):
        '''Returns a list of items with personalized promotions on them for a given user under the specified project deployment. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'item_code' mapped to mapping 'ITEM_ID' in our system).'''
        return self._call_api('getPersonalizedRanking', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'preserveRanks': preserve_ranks, 'scalingFactors': scaling_factors})

    def get_ranked_items(self, deployment_token: str, deployment_id: str, query_data: dict, preserve_ranks: list = [], scaling_factors: list = []):
        '''Returns a list of re-ranked items for a selected user when a list of items is required to be reranked according to the user's preferences. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'item_code' mapped to mapping 'ITEM_ID' in our system).'''
        return self._call_api('getRankedItems', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'preserveRanks': preserve_ranks, 'scalingFactors': scaling_factors})

    def get_related_items(self, deployment_token: str, deployment_id: str, query_data: dict, num_items: int = 50, page: int = 1, scaling_factors: list = [], restrict_items: list = [], exclude_items: list = []):
        '''Returns a list of related items for a given item under the specified project deployment. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'item_code' mapped to mapping 'ITEM_ID' in our system).'''
        return self._call_api('getRelatedItems', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'numItems': num_items, 'page': page, 'scalingFactors': scaling_factors, 'restrictItems': restrict_items, 'excludeItems': exclude_items})

    def create_batch_prediction(self, deployment_id: str, name: str = None, global_prediction_args: dict = None, explanations: bool = False, output_format: str = None, output_location: str = None, database_connector_id: str = None, database_output_config: dict = None, refresh_schedule: str = None):
        '''Creates a batch prediction task with the specified deployment ID, output location, and batch prediction job name.'''
        return self._call_api('createBatchPrediction', 'POST', query_params={'deploymentId': deployment_id}, body={'name': name, 'globalPredictionArgs': global_prediction_args, 'explanations': explanations, 'outputFormat': output_format, 'outputLocation': output_location, 'databaseConnectorId': database_connector_id, 'databaseOutputConfig': database_output_config, 'refreshSchedule': refresh_schedule}, parse_type=BatchPrediction)

    def start_batch_prediction(self, batch_prediction_id: str):
        '''Starts a batch prediction task with the specified deployment ID, output location, and batch prediction job name.'''
        return self._call_api('startBatchPrediction', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id}, parse_type=BatchPredictionVersion)

    def get_batch_prediction_result(self, batch_prediction_version: str):
        '''Returns a stream containing the batch prediction results'''
        return self._call_api('getBatchPredictionResult', 'GET', query_params={'batchPredictionVersion': batch_prediction_version}, streamable_response=True)

    def get_batch_prediction_connector_errors(self, batch_prediction_version: str):
        '''Returns a stream containing the batch prediction external connection error, if it exists'''
        return self._call_api('getBatchPredictionConnectorErrors', 'GET', query_params={'batchPredictionVersion': batch_prediction_version}, streamable_response=True)

    def list_batch_predictions(self, project_id: str):
        '''Retrieves a list for the batch prediction jobs for the specified deployment.'''
        return self._call_api('listBatchPredictions', 'GET', query_params={'projectId': project_id}, parse_type=BatchPrediction)

    def describe_batch_prediction(self, batch_prediction_id: str):
        '''Describe batch prediction Description'''
        return self._call_api('describeBatchPrediction', 'GET', query_params={'batchPredictionId': batch_prediction_id}, parse_type=BatchPrediction)

    def list_batch_prediction_versions(self, batch_prediction_id: str):
        ''''''
        return self._call_api('listBatchPredictionVersions', 'GET', query_params={'batchPredictionId': batch_prediction_id}, parse_type=BatchPredictionVersion)

    def describe_batch_prediction_version(self, batch_prediction_version: str):
        ''''''
        return self._call_api('describeBatchPredictionVersion', 'GET', query_params={'batchPredictionVersion': batch_prediction_version}, parse_type=BatchPredictionVersion)

    def update_batch_prediction(self, batch_prediction_id: str, deployment_id: str = None, global_prediction_args: dict = None, explanations: bool = None, output_format: str = None):
        ''''''
        return self._call_api('updateBatchPrediction', 'POST', query_params={'deploymentId': deployment_id}, body={'batchPredictionId': batch_prediction_id, 'globalPredictionArgs': global_prediction_args, 'explanations': explanations, 'outputFormat': output_format}, parse_type=BatchPrediction)

    def set_batch_prediction_file_connector_output(self, batch_prediction_id: str, output_format: str = None, output_location: str = None):
        ''''''
        return self._call_api('setBatchPredictionFileConnectorOutput', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id, 'outputFormat': output_format, 'outputLocation': output_location}, parse_type=BatchPrediction)

    def set_batch_prediction_database_connector_output(self, batch_prediction_id: str, database_connector_id: str = None, database_output_config: dict = None):
        ''''''
        return self._call_api('setBatchPredictionDatabaseConnectorOutput', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id, 'databaseConnectorId': database_connector_id, 'databaseOutputConfig': database_output_config}, parse_type=BatchPrediction)

    def set_batch_prediction_output_to_console(self, batch_prediction_id: str):
        ''''''
        return self._call_api('setBatchPredictionOutputToConsole', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id}, parse_type=BatchPrediction)

    def set_batch_prediction_dataset(self, batch_prediction_id: str, dataset_type: str, dataset_id: str = None):
        ''''''
        return self._call_api('setBatchPredictionDataset', 'POST', query_params={'datasetId': dataset_id}, body={'batchPredictionId': batch_prediction_id, 'datasetType': dataset_type}, parse_type=BatchPrediction)

    def set_batch_prediction_feature_group(self, batch_prediction_id: str, dataset_type: str, feature_group_id: str = None):
        ''''''
        return self._call_api('setBatchPredictionFeatureGroup', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id, 'datasetType': dataset_type, 'featureGroupId': feature_group_id}, parse_type=BatchPrediction)

    def delete_batch_prediction(self, batch_prediction_id: str):
        ''''''
        return self._call_api('deleteBatchPrediction', 'DELETE', query_params={'batchPredictionId': batch_prediction_id})

    def add_user_item_interaction(self, streaming_token: str, dataset_id: str, timestamp: int, user_id: str, item_id: list, event_type: str, additional_attributes: dict):
        '''Adds a user-item interaction record (data row) to a streaming dataset.'''
        return self._call_api('addUserItemInteraction', 'POST', query_params={'streamingToken': streaming_token, 'datasetId': dataset_id}, body={'timestamp': timestamp, 'userId': user_id, 'itemId': item_id, 'eventType': event_type, 'additionalAttributes': additional_attributes})

    def upsert_user_attributes(self, streaming_token: str, dataset_id: str, user_id: str, user_attributes: dict):
        '''Adds a user attributes record (data row) to a streaming dataset.

        Either the streaming dataset ID or the project ID is required.
        '''
        return self._call_api('upsertUserAttributes', 'POST', query_params={'streamingToken': streaming_token, 'datasetId': dataset_id}, body={'userId': user_id, 'userAttributes': user_attributes})

    def upsert_item_attributes(self, streaming_token: str, dataset_id: str, item_id: str, item_attributes: dict):
        '''Adds an item attributes record (data row) to a streaming dataset.

        Either the streaming dataset ID or the project ID is required.
        '''
        return self._call_api('upsertItemAttributes', 'POST', query_params={'streamingToken': streaming_token, 'datasetId': dataset_id}, body={'itemId': item_id, 'itemAttributes': item_attributes})

    def add_multiple_user_item_interactions(self, streaming_token: str, dataset_id: str, interactions: list):
        '''Adds a user-item interaction record (data row) to a streaming dataset.'''
        return self._call_api('addMultipleUserItemInteractions', 'POST', query_params={'streamingToken': streaming_token, 'datasetId': dataset_id}, body={'interactions': interactions})

    def upsert_multiple_user_attributes(self, streaming_token: str, dataset_id: str, upserts: list):
        '''Adds multiple user attributes records (data row) to a streaming dataset.

        The streaming dataset ID is required.
        '''
        return self._call_api('upsertMultipleUserAttributes', 'POST', query_params={'streamingToken': streaming_token, 'datasetId': dataset_id}, body={'upserts': upserts})

    def upsert_multiple_item_attributes(self, streaming_token: str, dataset_id: str, upserts: list):
        '''Adds multiple item attributes records (data row) to a streaming dataset.

        The streaming dataset ID is required.
        '''
        return self._call_api('upsertMultipleItemAttributes', 'POST', query_params={'streamingToken': streaming_token, 'datasetId': dataset_id}, body={'upserts': upserts})
