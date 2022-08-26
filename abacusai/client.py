import inspect
import io
import json
import logging
import os
import re
import tarfile
import tempfile
import time
from functools import lru_cache
from typing import Dict, List

import pandas as pd
import requests
from packaging import version
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from .algorithm import Algorithm
from .api_key import ApiKey
from .application_connector import ApplicationConnector
from .batch_prediction import BatchPrediction
from .batch_prediction_version import BatchPredictionVersion
from .custom_train_function_info import CustomTrainFunctionInfo
from .data_prep_logs import DataPrepLogs
from .database_connector import DatabaseConnector
from .dataset import Dataset
from .dataset_column import DatasetColumn
from .dataset_version import DatasetVersion
from .deployment import Deployment
from .deployment_auth_token import DeploymentAuthToken
from .drift_distributions import DriftDistributions
from .feature import Feature
from .feature_group import FeatureGroup
from .feature_group_export import FeatureGroupExport
from .feature_group_export_download_url import FeatureGroupExportDownloadUrl
from .feature_group_template import FeatureGroupTemplate
from .feature_group_version import FeatureGroupVersion
from .file_connector import FileConnector
from .file_connector_instructions import FileConnectorInstructions
from .file_connector_verification import FileConnectorVerification
from .function_logs import FunctionLogs
from .model import Model
from .model_metrics import ModelMetrics
from .model_monitor import ModelMonitor
from .model_monitor_summary import ModelMonitorSummary
from .model_monitor_summary_from_org import ModelMonitorSummaryFromOrg
from .model_monitor_version import ModelMonitorVersion
from .model_monitor_version_metric_data import ModelMonitorVersionMetricData
from .model_version import ModelVersion
from .modification_lock_info import ModificationLockInfo
from .organization_group import OrganizationGroup
from .prediction_metric import PredictionMetric
from .prediction_metric_version import PredictionMetricVersion
from .problem_type import ProblemType
from .project import Project
from .project_dataset import ProjectDataset
from .project_validation import ProjectValidation
from .refresh_pipeline_run import RefreshPipelineRun
from .refresh_policy import RefreshPolicy
from .resolved_feature_group_template import ResolvedFeatureGroupTemplate
from .return_class import AbstractApiClass
from .schema import Schema
from .streaming_auth_token import StreamingAuthToken
from .streaming_connector import StreamingConnector
from .training_config_options import TrainingConfigOptions
from .upload import Upload
from .upload_part import UploadPart
from .use_case import UseCase
from .use_case_requirements import UseCaseRequirements
from .user import User


DEFAULT_SERVER = 'https://api.abacus.ai'
INVALID_PANDAS_COLUMN_NAME_CHARACTERS = '[^A-Za-z0-9_]'


def _requests_retry_session(retries=5, backoff_factor=0.1, status_forcelist=(502, 504), session=None):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


@lru_cache(maxsize=None)
def _discover_service_url(service_discovery_url, client_version, deployment_id, deployment_token):
    from .cryptography import get_public_key, verify_response
    if not service_discovery_url:
        return None
    params = {}
    if deployment_id:
        params = {'deploymentId': deployment_id,
                  'deploymentToken': deployment_token}
    response = _requests_retry_session().get(service_discovery_url, headers={
        'clientVersion': client_version}, params=params)
    response_dict = response.json()

    verify_response(get_public_key(), response_dict)
    if deployment_id:
        return response_dict['prediction_url']
    else:
        return response_dict['api_url']


@lru_cache()
def _get_service_discovery_url():
    return os.getenv('ABACUS_SERVICE_DISCOVERY_URL')


class ClientOptions:
    """
    Options for configuring the ApiClient

    Args:
        exception_on_404 (bool): If true, will raise an exception on a 404 from the server, else will return None.
        server (str): The default server endpoint to use for API requests
    """

    def __init__(self, exception_on_404: bool = True, server: str = DEFAULT_SERVER):
        self.exception_on_404 = exception_on_404
        self.server = server


class ApiException(Exception):
    """
    Default ApiException raised by APIs

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
    """

    def __init__(self, message: str, http_status: int, exception: str = None):
        self.message = message
        self.http_status = http_status
        self.exception = exception or 'ApiException'

    def __str__(self):
        return f'{self.exception}({self.http_status}): {self.message}'


class BaseApiClient:
    """
    Abstract Base API Client

    Args:
        api_key (str): The api key to use as authentication to the server
        server (str): The base server url to use to send API requets to
        client_options (ClientOptions): Optional API client configurations
        skip_version_check (bool): If true, will skip checking the server's current API version on initializing the client
    """
    client_version = '0.36.19'

    def __init__(self, api_key: str = None, server: str = None, client_options: ClientOptions = None, skip_version_check: bool = False):
        self.api_key = api_key
        if self.api_key is None:
            self.api_key = os.getenv('ABACUS_API_KEY')
        self.web_version = None
        self.client_options = client_options or ClientOptions()
        self.server = server or self.client_options.server
        self.user = None
        self.service_discovery_url = _get_service_discovery_url()
        self.default_prediction_url = 'https://predict.api.abacus.ai' if self.server == DEFAULT_SERVER else self.server
        # Connection and version check
        if not skip_version_check:
            try:
                self.web_version = self._call_api(
                    'version', 'GET', server_override=DEFAULT_SERVER)
                if version.parse(self.web_version) > version.parse(self.client_version):
                    logging.warning(
                        'A new version of the Abacus.AI library is available')
                    logging.warning(
                        f'Current Version: {self.client_version} -> New Version: {self.web_version}')
            except Exception:
                logging.error(
                    f'Failed get the current API version from Abacus.AI ({DEFAULT_SERVER}/api/v0/version)')
        if api_key is not None:
            try:
                self.user = self._call_api('getUser', 'GET')
            except Exception:
                logging.error('Invalid API Key')

    def _clean_api_objects(self, obj):
        for key, val in (obj or {}).items():
            if isinstance(val, StreamingAuthToken):
                obj[key] = val.streaming_token
            elif isinstance(val, DeploymentAuthToken):
                obj[key] = val.deployment_token
            elif isinstance(val, AbstractApiClass):
                obj[key] = getattr(val, 'id', None)
            elif callable(val):
                try:
                    obj[key] = inspect.getsource(val)
                except OSError:
                    raise OSError(
                        f'Could not get source for function {key}. Please pass a stringified version of this function when the function is defined in a shell environment.')

    def _call_api(
            self, action, method, query_params=None,
            body=None, files=None, parse_type=None, streamable_response=False, server_override=None):
        headers = {'apiKey': self.api_key, 'clientVersion': self.client_version,
                   'User-Agent': f'python-abacusai-{self.client_version}'}
        url = (server_override or self.server) + '/api/v0/' + action
        self._clean_api_objects(query_params)
        self._clean_api_objects(body)
        if self.service_discovery_url:
            discovered_url = _discover_service_url(self.service_discovery_url, self.client_version, query_params.get(
                'deploymentId') if query_params else None, query_params.get('deploymentToken') if query_params else None)
            if discovered_url:
                url = discovered_url + '/api/' + action
        response = self._request(url, method, query_params=query_params,
                                 headers=headers, body=body, files=files, stream=streamable_response)

        result = None
        success = False
        error_message = None
        error_type = None
        json_data = None
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
        except Exception as e:
            logging.warn(
                f"_call_api caught an exception {e} in processing json_data {json_data}. API call url method body: {url} {method} '{json.dumps(body)}'")
            error_message = response.text
        if not success:
            if response.status_code == 504:
                error_message = 'Gateway timeout, please try again later'
            elif response.status_code > 502 and response.status_code not in (501, 503):
                error_message = 'Internal Server Error, please contact dev@abacus.ai for support'
            elif response.status_code == 404 and not self.client_options.exception_on_404:
                return None
            if response.headers and response.headers.get('x-request-id'):
                error_message += f". Request ID: {response.headers.get('x-request-id')}"
            raise ApiException(error_message, response.status_code, error_type)
        return result

    def _build_class(self, return_class, values):
        if values is None or values == {}:
            return None
        if isinstance(values, list):
            return [self._build_class(return_class, val) for val in values if val is not None]
        type_inputs = inspect.signature(return_class.__init__).parameters
        return return_class(self, **{key: val for key, val in values.items() if key in type_inputs})

    def _request(self, url, method, query_params=None, headers=None,
                 body=None, files=None, stream=False):
        if method == 'GET':
            return _requests_retry_session().get(url, params=query_params, headers=headers, stream=stream)
        elif method == 'POST':
            return _requests_retry_session().post(url, params=query_params, json=body, headers=headers, files=files, timeout=90)
        elif method == 'PUT':
            return _requests_retry_session().put(url, params=query_params, data=body, headers=headers, files=files, timeout=90)
        elif method == 'PATCH':
            return _requests_retry_session().patch(url, params=query_params, json=body, headers=headers, files=files, timeout=90)
        elif method == 'DELETE':
            return _requests_retry_session().delete(url, params=query_params, data=body, headers=headers)
        else:
            raise ValueError(
                'HTTP method must be `GET`, `POST`, `PATCH`, `PUT` or `DELETE`'
            )

    def _poll(self, obj, wait_states: set, delay: int = 5, timeout: int = 300, poll_args: dict = {}, status_field=None):
        start_time = time.time()
        while obj.get_status(**poll_args) in wait_states:
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f'Maximum wait time of {timeout}s exceeded')
            time.sleep(delay)
        return obj.refresh()

    def _upload_from_pandas(self, upload, df, clean_column_names=False) -> Dataset:
        if clean_column_names:
            df = df.rename(columns={col: re.sub(
                INVALID_PANDAS_COLUMN_NAME_CHARACTERS, '_', col) for col in df.columns})
        bad_column_names = [col for col in df.columns if bool(
            re.search(INVALID_PANDAS_COLUMN_NAME_CHARACTERS, col))]
        if bad_column_names:
            raise ValueError(
                f'The dataframe\'s Column(s): {bad_column_names} contain illegal characters. Please rename the columns such that they only contain alphanumeric characters and underscores.')
        with tempfile.TemporaryFile(mode='w+b') as parquet_out:
            df.to_parquet(parquet_out, index=all(
                index for index in df.index.names))
            parquet_out.seek(0)
            return upload.upload_file(parquet_out)

    def _upload_from_spark(self, upload, df) -> Dataset:
        with tempfile.TemporaryDirectory() as spark_out:
            df.write.format('parquet').mode('overwrite').save(spark_out)
            with tempfile.TemporaryFile(mode='w+b') as tar_out:
                with tarfile.open(fileobj=tar_out, mode='w') as tar_file:
                    for file_name in os.listdir(spark_out):
                        if file_name.endswith('.parquet'):
                            tar_file.add(os.path.join(
                                spark_out, file_name), arcname=file_name)
                tar_out.seek(0)
                return upload.upload_file(tar_out)


class ReadOnlyClient(BaseApiClient):
    """
    Abacus.AI Read Only API Client. Only contains GET methods

    Args:
        api_key (str): The api key to use as authentication to the server
        server (str): The base server url to use to send API requets to
        client_options (ClientOptions): Optional API client configurations
        skip_version_check (bool): If true, will skip checking the server's current API version on initializing the client
    """

    def list_api_keys(self) -> List[ApiKey]:
        """Lists all of the user's API keys the user's organization.

        Returns:
            ApiKey: List of API Keys for this user."""
        return self._call_api('listApiKeys', 'GET', query_params={}, parse_type=ApiKey)

    def list_organization_users(self) -> List[User]:
        """Retrieves a list of all users in the organization.

        This method will retrieve a list containing all the users in the organization. The list includes pending users who have been invited to the organization.


        Returns:
            User: Array of all of the users in the Organization"""
        return self._call_api('listOrganizationUsers', 'GET', query_params={}, parse_type=User)

    def describe_user(self) -> User:
        """Get the current user's information, such as their name, email, admin status, etc.

        Returns:
            User: Information about the current User"""
        return self._call_api('describeUser', 'GET', query_params={}, parse_type=User)

    def list_organization_groups(self) -> List[OrganizationGroup]:
        """Lists all Organizations Groups within this Organization

        Returns:
            OrganizationGroup: List of Groups in this Organization"""
        return self._call_api('listOrganizationGroups', 'GET', query_params={}, parse_type=OrganizationGroup)

    def describe_organization_group(self, organization_group_id: str) -> OrganizationGroup:
        """Returns the specific organization group passes in by the user.

        Args:
            organization_group_id (str): The unique ID of the organization group to that needs to be described.

        Returns:
            OrganizationGroup: Information about a specific Organization Group"""
        return self._call_api('describeOrganizationGroup', 'GET', query_params={'organizationGroupId': organization_group_id}, parse_type=OrganizationGroup)

    def list_use_cases(self) -> List[UseCase]:
        """Retrieves a list of all use cases with descriptions. Use the given mappings to specify a use case when needed.

        Returns:
            UseCase: A list of UseCase objects describing all the use cases addressed by the platform. For details, please refer to"""
        return self._call_api('listUseCases', 'GET', query_params={}, parse_type=UseCase)

    def describe_problem_type(self, problem_type: str) -> ProblemType:
        """

        Args:
            problem_type (str): 

        Returns:
            ProblemType: None"""
        return self._call_api('describeProblemType', 'GET', query_params={'problemType': problem_type}, parse_type=ProblemType)

    def describe_use_case_requirements(self, use_case: str) -> UseCaseRequirements:
        """This API call returns the feature requirements for a specified use case

        Args:
            use_case (str): This will contain the Enum String for the use case whose dataset requirements are needed.

        Returns:
            UseCaseRequirements: The feature requirements of the use case are returned. This includes all the feature group required for the use case along with their descriptions and feature mapping details."""
        return self._call_api('describeUseCaseRequirements', 'GET', query_params={'useCase': use_case}, parse_type=UseCaseRequirements)

    def describe_project(self, project_id: str) -> Project:
        """Returns a description of a project.

        Args:
            project_id (str): The unique project ID

        Returns:
            Project: The project description is returned."""
        return self._call_api('describeProject', 'GET', query_params={'projectId': project_id}, parse_type=Project)

    def list_projects(self, limit: int = 100, start_after_id: str = None) -> List[Project]:
        """Retrieves a list of all projects in the current organization.

        Args:
            limit (int): The max length of the list of projects.
            start_after_id (str): The ID of the project after which the list starts.

        Returns:
            Project: An array of all projects in the Organization the user is currently logged in to."""
        return self._call_api('listProjects', 'GET', query_params={'limit': limit, 'startAfterId': start_after_id}, parse_type=Project)

    def list_project_datasets(self, project_id: str) -> List[ProjectDataset]:
        """Retrieves all dataset(s) attached to a specified project. This API returns all attributes of each dataset, such as its name, type, and ID.

        Args:
            project_id (str): The unique ID associated with the project.

        Returns:
            ProjectDataset: An array representing all of the datasets attached to the project."""
        return self._call_api('listProjectDatasets', 'GET', query_params={'projectId': project_id}, parse_type=ProjectDataset)

    def get_schema(self, project_id: str, dataset_id: str) -> List[Schema]:
        """[DEPRECATED] Returns a schema given a specific dataset in a project. The schema of the dataset consists of the columns in the dataset, the data type of the column, and the column's column mapping.

        Args:
            project_id (str): The unique ID associated with the project.
            dataset_id (str): The unique ID associated with the dataset.

        Returns:
            Schema: An array of objects for each column in the specified dataset."""
        logging.warning(
            'This function (getSchema) is deprecated and will be removed in a future version. Use get_dataset_schema instead.')
        return self._call_api('getSchema', 'GET', query_params={'projectId': project_id, 'datasetId': dataset_id}, parse_type=Schema)

    def validate_project(self, project_id: str, feature_group_ids: list = None) -> ProjectValidation:
        """Validates that the specified project has all required feature group types for its use case and that all required feature columns are set.

        Args:
            project_id (str): The unique ID associated with the project.
            feature_group_ids (list): The feature group IDS to validate

        Returns:
            ProjectValidation: The project validation. If the specified project is missing required columns or feature groups, the response includes an array of objects for each missing required feature group and the missing required features in each feature group."""
        return self._call_api('validateProject', 'GET', query_params={'projectId': project_id, 'featureGroupIds': feature_group_ids}, parse_type=ProjectValidation)

    def get_feature_group_schema(self, feature_group_id: str, project_id: str = None) -> List[Feature]:
        """Returns a schema given a specific FeatureGroup in a project.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            project_id (str): The unique ID associated with the project.

        Returns:
            Feature: An array of objects for each column in the specified feature group."""
        return self._call_api('getFeatureGroupSchema', 'GET', query_params={'featureGroupId': feature_group_id, 'projectId': project_id}, parse_type=Feature)

    def describe_feature_group(self, feature_group_id: str) -> FeatureGroup:
        """Describe a Feature Group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.

        Returns:
            FeatureGroup: The feature group object."""
        return self._call_api('describeFeatureGroup', 'GET', query_params={'featureGroupId': feature_group_id}, parse_type=FeatureGroup)

    def describe_feature_group_by_table_name(self, table_name: str) -> FeatureGroup:
        """Describe a Feature Group by the feature group's table name

        Args:
            table_name (str): The unique table name of the Feature Group to lookup

        Returns:
            FeatureGroup: The Feature Group"""
        return self._call_api('describeFeatureGroupByTableName', 'GET', query_params={'tableName': table_name}, parse_type=FeatureGroup)

    def list_feature_groups(self, limit: int = 100, start_after_id: str = None, feature_group_template_id: str = None, is_including_detached_from_template: bool = False) -> List[FeatureGroup]:
        """Enlist all the feature groups associated with a project. A user needs to specify the unique project ID to fetch all attached feature groups.

        Args:
            limit (int): The the number of feature groups to be retrieved.
            start_after_id (str): An offset parameter to exclude all feature groups till a specified ID.
            feature_group_template_id (str): If specified, limit results to feature groups attached to this template id.
            is_including_detached_from_template (bool): When feature_group_template_id is specified, include feature groups that were detached from that template id.

        Returns:
            FeatureGroup: All the feature groups in the organization"""
        return self._call_api('listFeatureGroups', 'GET', query_params={'limit': limit, 'startAfterId': start_after_id, 'featureGroupTemplateId': feature_group_template_id, 'isIncludingDetachedFromTemplate': is_including_detached_from_template}, parse_type=FeatureGroup)

    def list_project_feature_groups(self, project_id: str, filter_feature_group_use: str = None) -> FeatureGroup:
        """List all the feature groups associated with a project

        Args:
            project_id (str): The unique ID associated with the project.
            filter_feature_group_use (str): The feature group use filter, when given as an argument, only allows feature groups in this project to be returned if they are of the given use.  DATA_WRANGLING,  TRAINING_INPUT,  BATCH_PREDICTION_INPUT,  BATCH_PREDICTION_OUTPUT

        Returns:
            FeatureGroup: All the Feature Groups in the Organization"""
        return self._call_api('listProjectFeatureGroups', 'GET', query_params={'projectId': project_id, 'filterFeatureGroupUse': filter_feature_group_use}, parse_type=FeatureGroup)

    def get_feature_group_version_export_download_url(self, feature_group_export_id: str) -> FeatureGroupExportDownloadUrl:
        """Get a link to download the feature group version.

        Args:
            feature_group_export_id (str): The Feature Group Export to get signed url for.

        Returns:
            FeatureGroupExportDownloadUrl: The FeatureGroupExportDownloadUrl instance, which contains the download URL and expiration time."""
        return self._call_api('getFeatureGroupVersionExportDownloadUrl', 'GET', query_params={'featureGroupExportId': feature_group_export_id}, parse_type=FeatureGroupExportDownloadUrl)

    def describe_feature_group_export(self, feature_group_export_id: str) -> FeatureGroupExport:
        """A feature group export

        Args:
            feature_group_export_id (str): The ID of the feature group export.

        Returns:
            FeatureGroupExport: The feature group export"""
        return self._call_api('describeFeatureGroupExport', 'GET', query_params={'featureGroupExportId': feature_group_export_id}, parse_type=FeatureGroupExport)

    def list_feature_group_exports(self, feature_group_id: str) -> List[FeatureGroupExport]:
        """Lists all of the feature group exports for a given feature group

        Args:
            feature_group_id (str): The ID of the feature group

        Returns:
            FeatureGroupExport: The feature group exports"""
        return self._call_api('listFeatureGroupExports', 'GET', query_params={'featureGroupId': feature_group_id}, parse_type=FeatureGroupExport)

    def list_feature_group_modifiers(self, feature_group_id: str) -> ModificationLockInfo:
        """To list users who can modify a feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.

        Returns:
            ModificationLockInfo: Modification lock status and groups and organizations added to the feature group."""
        return self._call_api('listFeatureGroupModifiers', 'GET', query_params={'featureGroupId': feature_group_id}, parse_type=ModificationLockInfo)

    def get_materialization_logs(self, feature_group_version: str, stdout: bool = False, stderr: bool = False) -> FunctionLogs:
        """Returns logs for materialized feature group version.

        Args:
            feature_group_version (str): The Feature Group instance to export
            stdout (bool):  Set True to get info logs
            stderr (bool):  Set True to get error logs

        Returns:
            FunctionLogs: A function logs."""
        return self._call_api('getMaterializationLogs', 'GET', query_params={'featureGroupVersion': feature_group_version, 'stdout': stdout, 'stderr': stderr}, parse_type=FunctionLogs)

    def list_feature_group_versions(self, feature_group_id: str, limit: int = 100, start_after_version: str = None) -> List[FeatureGroupVersion]:
        """Retrieves a list of all feature group versions for the specified feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            limit (int): The max length of the returned versions
            start_after_version (str): Results will start after this version

        Returns:
            FeatureGroupVersion: An array of feature group version."""
        return self._call_api('listFeatureGroupVersions', 'GET', query_params={'featureGroupId': feature_group_id, 'limit': limit, 'startAfterVersion': start_after_version}, parse_type=FeatureGroupVersion)

    def describe_feature_group_version(self, feature_group_version: str) -> FeatureGroupVersion:
        """Get a specific feature group version.

        Args:
            feature_group_version (str): The unique ID associated with the feature group version.

        Returns:
            FeatureGroupVersion: A feature group version."""
        return self._call_api('describeFeatureGroupVersion', 'GET', query_params={'featureGroupVersion': feature_group_version}, parse_type=FeatureGroupVersion)

    def describe_feature_group_template(self, feature_group_template_id: str) -> FeatureGroupTemplate:
        """Describe a Feature Group Template.

        Args:
            feature_group_template_id (str): The unique ID of a feature group template.

        Returns:
            FeatureGroupTemplate: The feature group template object."""
        return self._call_api('describeFeatureGroupTemplate', 'GET', query_params={'featureGroupTemplateId': feature_group_template_id}, parse_type=FeatureGroupTemplate)

    def list_feature_group_templates(self, limit: int = 100, start_after_id: str = None, feature_group_id: str = None, should_include_system_templates: bool = False) -> List[FeatureGroupTemplate]:
        """List feature group templates, optionally scoped by the feature group that created the templates.

        Args:
            limit (int): The maximum number of templates to be retrieved.
            start_after_id (str): An offset parameter to exclude all templates till the specified feature group template ID.
            feature_group_id (str): If specified, limit to templates created from this feature group.
            should_include_system_templates (bool): 

        Returns:
            FeatureGroupTemplate: All the feature groups in the organization, optionally limited by the feature group that created the template(s)."""
        return self._call_api('listFeatureGroupTemplates', 'GET', query_params={'limit': limit, 'startAfterId': start_after_id, 'featureGroupId': feature_group_id, 'shouldIncludeSystemTemplates': should_include_system_templates}, parse_type=FeatureGroupTemplate)

    def list_project_feature_group_templates(self, project_id: str, limit: int = 100, start_after_id: str = None, should_include_all_system_templates: bool = False) -> List[FeatureGroupTemplate]:
        """List feature group templates for feature groups associated with the project.

        Args:
            project_id (str): Limit to templates associated with this project, e.g. templates associated with feature groups in this project.
            limit (int): The maximum number of templates to be retrieved.
            start_after_id (str): An offset parameter to exclude all templates till the specified feature group template ID.
            should_include_all_system_templates (bool): 

        Returns:
            FeatureGroupTemplate: All the feature groups in the organization, optionally limited by the feature group that created the template(s)."""
        return self._call_api('listProjectFeatureGroupTemplates', 'GET', query_params={'projectId': project_id, 'limit': limit, 'startAfterId': start_after_id, 'shouldIncludeAllSystemTemplates': should_include_all_system_templates}, parse_type=FeatureGroupTemplate)

    def suggest_feature_group_template_for_feature_group(self, feature_group_id: str) -> FeatureGroupTemplate:
        """Suggest values for a feature gruop template, based on a feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group to use for suggesting values to use for the template.

        Returns:
            FeatureGroupTemplate: None"""
        return self._call_api('suggestFeatureGroupTemplateForFeatureGroup', 'GET', query_params={'featureGroupId': feature_group_id}, parse_type=FeatureGroupTemplate)

    def get_dataset_schema(self, dataset_id: str) -> List[DatasetColumn]:
        """Retrieves the column schema of a dataset

        Args:
            dataset_id (str): The Dataset schema to lookup.

        Returns:
            DatasetColumn: List of Column schema definitions"""
        return self._call_api('getDatasetSchema', 'GET', query_params={'datasetId': dataset_id}, parse_type=DatasetColumn)

    def get_file_connector_instructions(self, bucket: str, write_permission: bool = False) -> FileConnectorInstructions:
        """Retrieves verification information to create a data connector to a cloud storage bucket.

        Args:
            bucket (str): The fully qualified URI of the storage bucket to verify.
            write_permission (bool): If `true`, instructions will include steps for allowing Abacus.AI to write to this service.

        Returns:
            FileConnectorInstructions: An object with full description of the cloud storage bucket authentication options and bucket policy. Returns an error message if the parameters are invalid."""
        return self._call_api('getFileConnectorInstructions', 'GET', query_params={'bucket': bucket, 'writePermission': write_permission}, parse_type=FileConnectorInstructions)

    def list_database_connectors(self) -> DatabaseConnector:
        """Retrieves a list of all of the database connectors along with all their attributes.

        Returns:
            DatabaseConnector: The database Connector"""
        return self._call_api('listDatabaseConnectors', 'GET', query_params={}, parse_type=DatabaseConnector)

    def list_file_connectors(self) -> List[FileConnector]:
        """Retrieves a list of all connected services in the organization and their current verification status.

        Returns:
            FileConnector: An array of cloud storage buckets connected to the organization."""
        return self._call_api('listFileConnectors', 'GET', query_params={}, parse_type=FileConnector)

    def list_database_connector_objects(self, database_connector_id: str) -> List[str]:
        """Lists querable objects in the database connector.

        Args:
            database_connector_id (str): The unique identifier for the database connector."""
        return self._call_api('listDatabaseConnectorObjects', 'GET', query_params={'databaseConnectorId': database_connector_id})

    def get_database_connector_object_schema(self, database_connector_id: str, object_name: str = None) -> List[str]:
        """Get the schema of an object in an database connector.

        Args:
            database_connector_id (str): The unique identifier for the database connector.
            object_name (str): The unique identifier for the object in the external system."""
        return self._call_api('getDatabaseConnectorObjectSchema', 'GET', query_params={'databaseConnectorId': database_connector_id, 'objectName': object_name})

    def list_application_connectors(self) -> ApplicationConnector:
        """Retrieves a list of all of the application connectors along with all their attributes.

        Returns:
            ApplicationConnector: The appplication Connector"""
        return self._call_api('listApplicationConnectors', 'GET', query_params={}, parse_type=ApplicationConnector)

    def list_application_connector_objects(self, application_connector_id: str) -> List[str]:
        """Lists querable objects in the application connector.

        Args:
            application_connector_id (str): The unique identifier for the application connector."""
        return self._call_api('listApplicationConnectorObjects', 'GET', query_params={'applicationConnectorId': application_connector_id})

    def list_streaming_connectors(self) -> StreamingConnector:
        """Retrieves a list of all of the streaming connectors along with all their attributes.

        Returns:
            StreamingConnector: The streaming Connector"""
        return self._call_api('listStreamingConnectors', 'GET', query_params={}, parse_type=StreamingConnector)

    def list_streaming_tokens(self) -> List[StreamingAuthToken]:
        """Retrieves a list of all streaming tokens along with their attributes.

        Returns:
            StreamingAuthToken: An array of streaming tokens."""
        return self._call_api('listStreamingTokens', 'GET', query_params={}, parse_type=StreamingAuthToken)

    def get_recent_feature_group_streamed_data(self, feature_group_id: str):
        """Returns recently streamed data to a streaming feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group."""
        return self._call_api('getRecentFeatureGroupStreamedData', 'GET', query_params={'featureGroupId': feature_group_id})

    def list_uploads(self) -> List[Upload]:
        """Lists all ongoing uploads in the organization

        Returns:
            Upload: An array of uploads."""
        return self._call_api('listUploads', 'GET', query_params={}, parse_type=Upload)

    def describe_upload(self, upload_id: str) -> Upload:
        """Retrieves the current upload status (complete or inspecting) and the list of file parts uploaded for a specified dataset upload.

        Args:
            upload_id (str): The unique ID associated with the file uploaded or being uploaded in parts.

        Returns:
            Upload: The details associated with the large dataset file uploaded in parts."""
        return self._call_api('describeUpload', 'GET', query_params={'uploadId': upload_id}, parse_type=Upload)

    def list_datasets(self, limit: int = 100, start_after_id: str = None, exclude_streaming: bool = False) -> List[Dataset]:
        """Retrieves a list of all of the datasets in the organization.

        Args:
            limit (int): The max length of the list of projects.
            start_after_id (str): The ID of the project after which the list starts.
            exclude_streaming (bool): Exclude streaming datasets from result.

        Returns:
            Dataset: A list of datasets."""
        return self._call_api('listDatasets', 'GET', query_params={'limit': limit, 'startAfterId': start_after_id, 'excludeStreaming': exclude_streaming}, parse_type=Dataset)

    def describe_dataset(self, dataset_id: str) -> Dataset:
        """Retrieves a full description of the specified dataset, with attributes such as its ID, name, source type, etc.

        Args:
            dataset_id (str): The unique ID associated with the dataset.

        Returns:
            Dataset: The dataset."""
        return self._call_api('describeDataset', 'GET', query_params={'datasetId': dataset_id}, parse_type=Dataset)

    def describe_dataset_version(self, dataset_version: str) -> DatasetVersion:
        """Retrieves a full description of the specified dataset version, with attributes such as its ID, name, source type, etc.

        Args:
            dataset_version (str): The unique ID associated with the dataset version.

        Returns:
            DatasetVersion: The dataset version."""
        return self._call_api('describeDatasetVersion', 'GET', query_params={'datasetVersion': dataset_version}, parse_type=DatasetVersion)

    def list_dataset_versions(self, dataset_id: str, limit: int = 100, start_after_version: str = None) -> List[DatasetVersion]:
        """Retrieves a list of all dataset versions for the specified dataset.

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            limit (int): The max length of the list of all dataset versions.
            start_after_version (str): The id of the version after which the list starts.

        Returns:
            DatasetVersion: A list of dataset versions."""
        return self._call_api('listDatasetVersions', 'GET', query_params={'datasetId': dataset_id, 'limit': limit, 'startAfterVersion': start_after_version}, parse_type=DatasetVersion)

    def get_training_config_options(self, project_id: str, feature_group_ids: list = None, for_retrain: bool = False) -> List[TrainingConfigOptions]:
        """Retrieves the full description of the model training configuration options available for the specified project.

        The configuration options available are determined by the use case associated with the specified project. Refer to the (Use Case Documentation)[https://api.abacus.ai/app/help/useCases] for more information on use cases and use case specific configuration options.


        Args:
            project_id (str): The unique ID associated with the project.
            feature_group_ids (list): The feature group IDs to be used for training
            for_retrain (bool): If training config options are used for retrain

        Returns:
            TrainingConfigOptions: An array of options that can be specified when training a model in this project."""
        return self._call_api('getTrainingConfigOptions', 'GET', query_params={'projectId': project_id, 'featureGroupIds': feature_group_ids, 'forRetrain': for_retrain}, parse_type=TrainingConfigOptions)

    def list_models(self, project_id: str) -> List[Model]:
        """Retrieves the list of models in the specified project.

        Args:
            project_id (str): The unique ID associated with the project.

        Returns:
            Model: An array of models."""
        return self._call_api('listModels', 'GET', query_params={'projectId': project_id}, parse_type=Model)

    def describe_model(self, model_id: str) -> Model:
        """Retrieves a full description of the specified model.

        Args:
            model_id (str): The unique ID associated with the model.

        Returns:
            Model: The description of the model."""
        return self._call_api('describeModel', 'GET', query_params={'modelId': model_id}, parse_type=Model)

    def get_model_metrics(self, model_id: str, model_version: str = None, baseline_metrics: bool = False) -> ModelMetrics:
        """Retrieves a full list of the metrics for the specified model.

        If only the model's unique identifier (modelId) is specified, the latest trained version of model (modelVersion) is used.


        Args:
            model_id (str): The unique ID associated with the model.
            model_version (str): The version of the model.
            baseline_metrics (bool): If true, will also return the baseline model metrics for comparison.

        Returns:
            ModelMetrics: An object to show the model metrics and explanations for what each metric means."""
        return self._call_api('getModelMetrics', 'GET', query_params={'modelId': model_id, 'modelVersion': model_version, 'baselineMetrics': baseline_metrics}, parse_type=ModelMetrics)

    def list_model_versions(self, model_id: str, limit: int = 100, start_after_version: str = None) -> List[ModelVersion]:
        """Retrieves a list of the version for a given model.

        Args:
            model_id (str): The unique ID associated with the model.
            limit (int): The max length of the list of all dataset versions.
            start_after_version (str): The id of the version after which the list starts.

        Returns:
            ModelVersion: An array of model versions."""
        return self._call_api('listModelVersions', 'GET', query_params={'modelId': model_id, 'limit': limit, 'startAfterVersion': start_after_version}, parse_type=ModelVersion)

    def describe_model_version(self, model_version: str) -> ModelVersion:
        """Retrieves a full description of the specified model version

        Args:
            model_version (str): The unique version ID of the model version

        Returns:
            ModelVersion: A model version."""
        return self._call_api('describeModelVersion', 'GET', query_params={'modelVersion': model_version}, parse_type=ModelVersion)

    def get_training_data_logs(self, model_version: str) -> DataPrepLogs:
        """Retrieves the data preparation logs during model training.

        Args:
            model_version (str): The unique version ID of the model version

        Returns:
            DataPrepLogs: A list of logs."""
        return self._call_api('getTrainingDataLogs', 'GET', query_params={'modelVersion': model_version}, parse_type=DataPrepLogs)

    def set_default_model_algorithm(self, model_id: str = None, algorithm: str = None):
        """Sets the model's algorithm to default for all new deployments

        Args:
            model_id (Unique String Identifier): The model to set
            algorithm (Enum String): the algorithm to pin in the model


        Args:
            model_id (str): 
            algorithm (str): """
        return self._call_api('setDefaultModelAlgorithm', 'GET', query_params={'modelId': model_id, 'algorithm': algorithm})

    def get_training_logs(self, model_version: str, stdout: bool = False, stderr: bool = False) -> FunctionLogs:
        """Returns training logs for the model.

        Args:
            model_version (str): The unique version ID of the model version
            stdout (bool):  Set True to get info logs
            stderr (bool):  Set True to get error logs

        Returns:
            FunctionLogs: A function logs."""
        return self._call_api('getTrainingLogs', 'GET', query_params={'modelVersion': model_version, 'stdout': stdout, 'stderr': stderr}, parse_type=FunctionLogs)

    def export_model_artifacts(self, model_version: str) -> io.BytesIO:
        """Returns model artifacts zip.

        Args:
            model_version (str): The version of the model."""
        return self._call_api('exportModelArtifacts', 'GET', query_params={'modelVersion': model_version}, streamable_response=True)

    def list_model_monitors(self, project_id: str) -> List[ModelMonitor]:
        """Retrieves the list of models monitors in the specified project.

        Args:
            project_id (str): The unique ID associated with the project.

        Returns:
            ModelMonitor: An array of model monitors."""
        return self._call_api('listModelMonitors', 'GET', query_params={'projectId': project_id}, parse_type=ModelMonitor)

    def describe_model_monitor(self, model_monitor_id: str) -> ModelMonitor:
        """Retrieves a full description of the specified model monitor.

        Args:
            model_monitor_id (str): The unique ID associated with the model monitor.

        Returns:
            ModelMonitor: The description of the model monitor."""
        return self._call_api('describeModelMonitor', 'GET', query_params={'modelMonitorId': model_monitor_id}, parse_type=ModelMonitor)

    def get_prediction_drift(self, model_monitor_version: str) -> DriftDistributions:
        """Gets the label and prediction drifts for a model monitor.

        Args:
            model_monitor_version (str): The unique identifier to a model monitor version created under the project.

        Returns:
            DriftDistributions: An object describing training and prediction output label and prediction distributions."""
        return self._call_api('getPredictionDrift', 'GET', query_params={'modelMonitorVersion': model_monitor_version}, parse_type=DriftDistributions)

    def get_model_monitor_summary(self, model_monitor_id: str) -> ModelMonitorSummary:
        """Gets the summary of a model monitor across versions.

        Args:
            model_monitor_id (str): The unique ID associated with the model monitor.

        Returns:
            ModelMonitorSummary: An object describing integrity, bias violations, model accuracy, and drift for a model monitor."""
        return self._call_api('getModelMonitorSummary', 'GET', query_params={'modelMonitorId': model_monitor_id}, parse_type=ModelMonitorSummary)

    def list_model_monitor_versions(self, model_monitor_id: str, limit: int = 100, start_after_version: str = None) -> List[ModelMonitorVersion]:
        """Retrieves a list of the versions for a given model monitor.

        Args:
            model_monitor_id (str): The unique ID associated with the model monitor.
            limit (int): The max length of the list of all model monitor versions.
            start_after_version (str): The id of the version after which the list starts.

        Returns:
            ModelMonitorVersion: An array of model monitor versions."""
        return self._call_api('listModelMonitorVersions', 'GET', query_params={'modelMonitorId': model_monitor_id, 'limit': limit, 'startAfterVersion': start_after_version}, parse_type=ModelMonitorVersion)

    def describe_model_monitor_version(self, model_monitor_version: str) -> ModelMonitorVersion:
        """Retrieves a full description of the specified model monitor version

        Args:
            model_monitor_version (str): The unique version ID of the model monitor version

        Returns:
            ModelMonitorVersion: A model monitor version."""
        return self._call_api('describeModelMonitorVersion', 'GET', query_params={'modelMonitorVersion': model_monitor_version}, parse_type=ModelMonitorVersion)

    def model_monitor_version_metric_data(self, model_monitor_version: str, metric_type: str) -> ModelMonitorVersionMetricData:
        """Provides the data needed for decile metrics associated with the model monitor.

        Args:
            model_monitor_version (str): Model monitor version id.
            metric_type (str): The metric type to get data for.

        Returns:
            ModelMonitorVersionMetricData: Data associated with the metric."""
        return self._call_api('modelMonitorVersionMetricData', 'GET', query_params={'modelMonitorVersion': model_monitor_version, 'metricType': metric_type}, parse_type=ModelMonitorVersionMetricData)

    def get_model_monitor_chart_from_organization(self, organization_id: str, chart_type: str, limit: int = 15) -> List[ModelMonitorSummaryFromOrg]:
        """Gets a list of model monitor summaries across monitors for an organization.

        Args:
            organization_id (str): The unique ID associated with the organization.
            chart_type (str): The type of chart (model_accuracy, bias_violations, data_integrity, or model_drift) to return.
            limit (int): The max length of the model monitors.

        Returns:
            ModelMonitorSummaryFromOrg: A list of ModelMonitorSummaryForOrganization objects describing accuracy, bias, drift, or integrity for all model monitors in an organization."""
        return self._call_api('getModelMonitorChartFromOrganization', 'GET', query_params={'organizationId': organization_id, 'chartType': chart_type, 'limit': limit}, parse_type=ModelMonitorSummaryFromOrg)

    def get_model_monitoring_logs(self, model_monitor_version: str, stdout: bool = False, stderr: bool = False) -> FunctionLogs:
        """Returns monitoring logs for the model.

        Args:
            model_monitor_version (str): The unique version ID of the model monitor version
            stdout (bool):  Set True to get info logs
            stderr (bool):  Set True to get error logs

        Returns:
            FunctionLogs: A function logs."""
        return self._call_api('getModelMonitoringLogs', 'GET', query_params={'modelMonitorVersion': model_monitor_version, 'stdout': stdout, 'stderr': stderr}, parse_type=FunctionLogs)

    def get_drift_for_feature(self, model_monitor_version: str, feature_name: str) -> Dict:
        """Gets the feature drift associated with a single feature in an output feature group from a prediction.

        Args:
            model_monitor_version (str): The unique identifier to a model monitor version created under the project.
            feature_name (str): Name of the feature to view the distribution of."""
        return self._call_api('getDriftForFeature', 'GET', query_params={'modelMonitorVersion': model_monitor_version, 'featureName': feature_name})

    def get_outliers_for_feature(self, model_monitor_version: str, feature_name: str = None) -> Dict:
        """Gets a list of outliers measured by a single feature (or overall) in an output feature group from a prediction.

        Args:
            model_monitor_version (str): The unique identifier to a model monitor version created under the project.
            feature_name (str): Name of the feature to view the distribution of."""
        return self._call_api('getOutliersForFeature', 'GET', query_params={'modelMonitorVersion': model_monitor_version, 'featureName': feature_name})

    def get_outliers_for_batch_prediction_feature(self, batch_prediction_version: str, feature_name: str = None) -> Dict:
        """Gets a list of outliers measured by a single feature (or overall) in an output feature group from a prediction.

        Args:
            batch_prediction_version (str): The unique identifier to a batch prediction version created under the project.
            feature_name (str): Name of the feature to view the distribution of."""
        return self._call_api('getOutliersForBatchPredictionFeature', 'GET', query_params={'batchPredictionVersion': batch_prediction_version, 'featureName': feature_name})

    def describe_deployment(self, deployment_id: str) -> Deployment:
        """Retrieves a full description of the specified deployment.

        Args:
            deployment_id (str): The unique ID associated with the deployment.

        Returns:
            Deployment: The description of the deployment."""
        return self._call_api('describeDeployment', 'GET', query_params={'deploymentId': deployment_id}, parse_type=Deployment)

    def list_deployments(self, project_id: str) -> List[Deployment]:
        """Retrieves a list of all deployments in the specified project.

        Args:
            project_id (str): The unique ID associated with the project.

        Returns:
            Deployment: An array of deployments."""
        return self._call_api('listDeployments', 'GET', query_params={'projectId': project_id}, parse_type=Deployment)

    def list_deployment_tokens(self, project_id: str) -> List[DeploymentAuthToken]:
        """Retrieves a list of all deployment tokens in the specified project.

        Args:
            project_id (str): The unique ID associated with the project.

        Returns:
            DeploymentAuthToken: An array of deployment tokens."""
        return self._call_api('listDeploymentTokens', 'GET', query_params={'projectId': project_id}, parse_type=DeploymentAuthToken)

    def describe_refresh_policy(self, refresh_policy_id: str) -> RefreshPolicy:
        """Retrieve a single refresh policy

        Args:
            refresh_policy_id (str): The unique ID associated with this refresh policy

        Returns:
            RefreshPolicy: A refresh policy object"""
        return self._call_api('describeRefreshPolicy', 'GET', query_params={'refreshPolicyId': refresh_policy_id}, parse_type=RefreshPolicy)

    def describe_refresh_pipeline_run(self, refresh_pipeline_run_id: str) -> RefreshPipelineRun:
        """Retrieve a single refresh pipeline run

        Args:
            refresh_pipeline_run_id (str): The unique ID associated with this refresh pipeline_run

        Returns:
            RefreshPipelineRun: A refresh pipeline run object"""
        return self._call_api('describeRefreshPipelineRun', 'GET', query_params={'refreshPipelineRunId': refresh_pipeline_run_id}, parse_type=RefreshPipelineRun)

    def list_refresh_policies(self, project_id: str = None, dataset_ids: list = [], model_ids: list = [], deployment_ids: list = [], batch_prediction_ids: list = [], model_monitor_ids: list = [], prediction_metric_ids: list = []) -> RefreshPolicy:
        """List the refresh policies for the organization

        Args:
            project_id (str): Optionally, a Project ID can be specified so that all datasets, models and deployments are captured at the instant this policy was created
            dataset_ids (list): Comma separated list of Dataset IDs
            model_ids (list): Comma separated list of Model IDs
            deployment_ids (list): Comma separated list of Deployment IDs
            batch_prediction_ids (list): Comma separated list of Batch Prediction IDs
            model_monitor_ids (list): Comma separated list of Model Monitor IDs.
            prediction_metric_ids (list): Comma separated list of Prediction Metric IDs,

        Returns:
            RefreshPolicy: List of all refresh policies in the organization"""
        return self._call_api('listRefreshPolicies', 'GET', query_params={'projectId': project_id, 'datasetIds': dataset_ids, 'modelIds': model_ids, 'deploymentIds': deployment_ids, 'batchPredictionIds': batch_prediction_ids, 'modelMonitorIds': model_monitor_ids, 'predictionMetricIds': prediction_metric_ids}, parse_type=RefreshPolicy)

    def list_refresh_pipeline_runs(self, refresh_policy_id: str) -> RefreshPipelineRun:
        """List the the times that the refresh policy has been run

        Args:
            refresh_policy_id (str): The unique ID associated with this refresh policy

        Returns:
            RefreshPipelineRun: A list of refresh pipeline runs for the given refresh policy id"""
        return self._call_api('listRefreshPipelineRuns', 'GET', query_params={'refreshPolicyId': refresh_policy_id}, parse_type=RefreshPipelineRun)

    def list_prediction_metrics(self, feature_group_id: str, limit: int = 100, should_include_latest_version_description: bool = True, start_after_id: str = None) -> List[PredictionMetric]:
        """List the prediction metrics for a feature group.

        Args:
            feature_group_id (str): The feature group used as input to this prediction metric.
            limit (int): The the number of prediction metrics to be retrieved.
            should_include_latest_version_description (bool): include the description of the latest prediction metric version for each prediction metric
            start_after_id (str): An offset parameter to exclude all prediction metrics till the specified prediction metric ID.

        Returns:
            PredictionMetric: The prediction metrics for this feature group."""
        return self._call_api('listPredictionMetrics', 'GET', query_params={'featureGroupId': feature_group_id, 'limit': limit, 'shouldIncludeLatestVersionDescription': should_include_latest_version_description, 'startAfterId': start_after_id}, parse_type=PredictionMetric)

    def query_prediction_metrics(self, feature_group_id: str = None, project_id: str = None, limit: int = 100, should_include_latest_version_description: bool = True, start_after_id: str = None) -> List[PredictionMetric]:
        """Query and return prediction metrics and extra data needed by the UI, constrained by the parameters provided.

        feature_group_id (Unique String Identifier): [optional] The feature group used as input to the prediction metrics.
            project_id (Unique String Identifier): [optional] The project_id of the prediction metrics.
            limit (Integer): The the number of prediction metrics to be retrieved.
            should_include_latest_version_description (Boolean): include the description of the latest prediction metric version for each prediction metric
            start_after_id (Unique String Identifier): An offset parameter to exclude all prediction metrics till the specified prediction metric ID.


        Args:
            feature_group_id (str): 
            project_id (str): 
            limit (int): 
            should_include_latest_version_description (bool): 
            start_after_id (str): 

        Returns:
            PredictionMetric: The prediction metrics for this feature group."""
        return self._call_api('queryPredictionMetrics', 'GET', query_params={'featureGroupId': feature_group_id, 'projectId': project_id, 'limit': limit, 'shouldIncludeLatestVersionDescription': should_include_latest_version_description, 'startAfterId': start_after_id}, parse_type=PredictionMetric)

    def describe_prediction_metric_version(self, prediction_metric_version: str) -> PredictionMetricVersion:
        """Retrieves a full description of the specified prediction metric version

        Args:
            prediction_metric_version (str): The unique version ID of the prediction metric version

        Returns:
            PredictionMetricVersion: A prediction metric version. For more information, please refer to the details on the object (below)."""
        return self._call_api('describePredictionMetricVersion', 'GET', query_params={'predictionMetricVersion': prediction_metric_version}, parse_type=PredictionMetricVersion)

    def download_batch_prediction_result_chunk(self, batch_prediction_version: str, offset: int = 0, chunk_size: int = 10485760) -> io.BytesIO:
        """Returns a stream containing the batch prediction results

        Args:
            batch_prediction_version (str): The unique identifier of the batch prediction version to get the results from
            offset (int): The offset to read from
            chunk_size (int): The max amount of data to read"""
        return self._call_api('downloadBatchPredictionResultChunk', 'GET', query_params={'batchPredictionVersion': batch_prediction_version, 'offset': offset, 'chunkSize': chunk_size}, streamable_response=True)

    def get_batch_prediction_connector_errors(self, batch_prediction_version: str) -> io.BytesIO:
        """Returns a stream containing the batch prediction database connection write errors, if any writes failed to the database connector

        Args:
            batch_prediction_version (str): The unique identifier of the batch prediction job to get the errors for"""
        return self._call_api('getBatchPredictionConnectorErrors', 'GET', query_params={'batchPredictionVersion': batch_prediction_version}, streamable_response=True)

    def list_batch_predictions(self, project_id: str) -> List[BatchPrediction]:
        """Retrieves a list for the batch predictions in the project

        Args:
            project_id (str): The unique identifier of the project

        Returns:
            BatchPrediction: A list of batch prediction jobs."""
        return self._call_api('listBatchPredictions', 'GET', query_params={'projectId': project_id}, parse_type=BatchPrediction)

    def describe_batch_prediction(self, batch_prediction_id: str) -> BatchPrediction:
        """Describes the batch prediction

        Args:
            batch_prediction_id (str): The unique ID associated with the batch prediction.

        Returns:
            BatchPrediction: The batch prediction description."""
        return self._call_api('describeBatchPrediction', 'GET', query_params={'batchPredictionId': batch_prediction_id}, parse_type=BatchPrediction)

    def list_batch_prediction_versions(self, batch_prediction_id: str, limit: int = 100, start_after_version: str = None) -> List[BatchPredictionVersion]:
        """Retrieves a list of versions of a given batch prediction

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction
            limit (int): The number of versions to list
            start_after_version (str): The version to start after

        Returns:
            BatchPredictionVersion: A list of batch prediction versions."""
        return self._call_api('listBatchPredictionVersions', 'GET', query_params={'batchPredictionId': batch_prediction_id, 'limit': limit, 'startAfterVersion': start_after_version}, parse_type=BatchPredictionVersion)

    def describe_batch_prediction_version(self, batch_prediction_version: str) -> BatchPredictionVersion:
        """Describes a batch prediction version

        Args:
            batch_prediction_version (str): The unique identifier of the batch prediction version

        Returns:
            BatchPredictionVersion: The batch prediction version."""
        return self._call_api('describeBatchPredictionVersion', 'GET', query_params={'batchPredictionVersion': batch_prediction_version}, parse_type=BatchPredictionVersion)

    def describe_algorithm(self, algorithm: str) -> Algorithm:
        """Retrieves a full description of the specified algorithm.

        Args:
            algorithm (str): The name of the algorithm.

        Returns:
            Algorithm: The description of the Algorithm."""
        return self._call_api('describeAlgorithm', 'GET', query_params={'algorithm': algorithm}, parse_type=Algorithm)


def get_source_code_info(train_function: callable, predict_function: callable = None, predict_many_function: callable = None, initialize_function: callable = None):
    if not train_function:
        return None, None, None, None, None
    function_source_code = inspect.getsource(train_function)
    predict_function_name, predict_many_function_name, initialize_function_name = None, None, None
    if predict_function is not None:
        predict_function_name = predict_function.__name__
        function_source_code = function_source_code + \
            '\n\n' + inspect.getsource(predict_function)
    if predict_many_function is not None:
        predict_many_function_name = predict_many_function.__name__
        function_source_code = function_source_code + \
            '\n\n' + inspect.getsource(predict_many_function)
    if initialize_function is not None:
        initialize_function_name = initialize_function.__name__
        function_source_code = function_source_code + \
            '\n\n' + inspect.getsource(initialize_function)
    return function_source_code, train_function.__name__, predict_function_name, predict_many_function_name, initialize_function_name


class ApiClient(ReadOnlyClient):
    """
    Abacus.AI API Client

    Args:
        api_key (str): The api key to use as authentication to the server
        server (str): The base server url to use to send API requets to
        client_options (ClientOptions): Optional API client configurations
        skip_version_check (bool): If true, will skip checking the server's current API version on initializing the client
    """

    def create_dataset_from_pandas(self, feature_group_table_name: str, df: pd.DataFrame, name: str = None, clean_column_names: bool = False) -> Dataset:
        """
        [Deprecated]
        Creates a Dataset from a pandas dataframe

        Args:
            feature_group_table_name (str): The table name to assign to the feature group created by this call
            df (pandas.DataFrame): The dataframe to upload
            name (str): The name to give to the dataset

        Returns:
            Dataset: The dataset object created
        """
        upload = self.create_dataset_from_upload(
            name=name or feature_group_table_name, table_name=feature_group_table_name, file_format='PARQUET')
        return self._upload_from_pandas(upload, df, clean_column_names)

    def create_dataset_version_from_pandas(self, table_name_or_id: str, df: pd.DataFrame, clean_column_names: bool = False) -> Dataset:
        """
        [Deprecated]
        Updates an existing dataset from a pandas dataframe

        Args:
            table_name_or_id (str): The table name of the feature group or the ID of the dataset to update
            df (pandas.DataFrame): The dataframe to upload

        Returns:
            Dataset: The dataset updated
        """
        dataset_id = None
        try:
            self.describe_dataset(table_name_or_id)
            dataset_id = table_name_or_id
        except ApiException:
            pass
        if not dataset_id:
            feature_group = self.describe_feature_group_by_table_name(
                table_name_or_id)
            if feature_group.feature_group_source_type != 'DATASET':
                raise ApiException(
                    'Feature Group is not source type DATASET', 409, 'ConflictError')
            dataset_id = feature_group.dataset_id
        upload = self.create_dataset_version_from_upload(
            dataset_id, file_format='PARQUET')
        return self._upload_from_pandas(upload, df, clean_column_names)

    def create_feature_group_from_pandas_df(self, table_name: str, df) -> FeatureGroup:
        """Create a Feature Group from a local Pandas DataFrame.

        Args:
            table_name (str): The table name to assign to the feature group created by this call
            df (pandas.DataFrame): The dataframe to upload
        """
        dataset = self.create_dataset_from_pandas(
            feature_group_table_name=table_name, df=df)
        return dataset.describe_feature_group()

    def update_feature_group_from_pandas_df(self, table_name: str, df) -> FeatureGroup:
        """Updates a DATASET Feature Group from a local Pandas DataFrame.

        Args:
            table_name (str): The table name to assign to the feature group created by this call
            df (pandas.DataFrame): The dataframe to upload
        """
        feature_group = self.describe_feature_group_by_table_name(table_name)
        if feature_group.feature_group_source_type != 'DATASET':
            raise ApiException(
                'Feature Group is not source type DATASET', 409, 'ConflictError')
        dataset_id = feature_group.dataset_id
        upload = self.create_dataset_version_from_upload(
            dataset_id, file_format='PARQUET')
        return self._upload_from_df(upload, df).describe_feature_group()

    def create_feature_group_from_spark_df(self, table_name: str, df) -> FeatureGroup:
        """Create a Feature Group from a local Spark DataFrame.

        Args:
            df (pyspark.sql.DataFrame): The dataframe to upload
            table_name (str): The table name to assign to the feature group created by this call
        """

        upload = self.create_dataset_from_upload(
            name=table_name, table_name=table_name, file_format='TAR')
        return self._upload_from_spark(upload, df).describe_feature_group()

    def update_feature_group_from_spark_df(self, table_name: str, df) -> FeatureGroup:
        """Create a Feature Group from a local Spark DataFrame.

        Args:
            df (pyspark.sql.DataFrame): The dataframe to upload
            table_name (str): The table name to assign to the feature group created by this call
            should_wait_for_upload (bool): Wait for dataframe to upload before returning. Some FeatureGroup methods, like materialization, may not work until upload is complete.
            timeout (int, optional): If waiting for upload, time out after this limit.
        """
        feature_group = self.describe_feature_group_by_table_name(table_name)
        if feature_group.feature_group_source_type != 'DATASET':
            raise ApiException(
                'Feature Group is not source type DATASET', 409, 'ConflictError')
        dataset_id = feature_group.dataset_id
        upload = self.create_dataset_version_from_upload(
            dataset_id, file_format='TAR')
        return self._upload_from_spark(upload, df).describe_feature_group()

    def create_spark_df_from_feature_group_version(self, session, feature_group_version: str):
        """Create a Spark Dataframe in the provided Spark Session context, for a materialized Abacus Feature Group Version.

        Args:
            session (pyspark.sql.SparkSession): Spark session
            feature_group_version (str): Feature group version to load from

        Returns:
            pyspark.sql.DataFrame
        """
        feature_group_version_object = self.describe_feature_group_version(
            feature_group_version)
        feature_group_version_object.wait_for_results()
        # TODO: if waited-for results not successful, return informaive error
        feature_group_version_pandas_df = feature_group_version_object.load_as_pandas()
        return session.createDataFrame(feature_group_version_pandas_df)

    def create_model_from_functions(self, project_id: str, train_function: callable, predict_function: callable = None, training_input_tables: list = None, predict_many_function: callable = None, initialize_function: callable = None, cpu_size: str = None, memory: int = None, training_config: dict = None, exclusive_run: bool = False):
        """
        Creates a model from a python function

        Args:
            project_id (str): The project to create the model in
            train_function (callable): The training fucntion callable to serialize and upload
            predict_function (callable): The predict function callable to serialize and upload
            predict_many_function (callable): The predict many function callable to serialize and upload
            initialize_function (callable): The initialize function callable to serialize and upload
            training_input_tables (list): The input table names of the feature groups to pass to the train function
            cpu_size (str): Size of the cpu for the training function
            memory (int): Memory (in GB) for the training function
        """
        function_source_code, train_function_name, predict_function_name, predict_many_function_name, initialize_function_name = get_source_code_info(
            train_function, predict_function, predict_many_function, initialize_function)
        return self.create_model_from_python(project_id=project_id, function_source_code=function_source_code, train_function_name=train_function_name, predict_function_name=predict_function_name, predict_many_function_name=predict_many_function_name, initialize_function_name=initialize_function_name, training_input_tables=training_input_tables, training_config=training_config, cpu_size=cpu_size, memory=memory, exclusive_run=exclusive_run)

    def create_feature_group_from_python_function(self, function: callable, table_name: str, input_tables: list, cpu_size: str = None, memory: int = None):
        """
        Creates a feature group from a python function

        Args:
            function (callable): The function callable for the feature group
            table_name (str): The table name to give the feature group
            input_tables (list): The input table names of the feature groups as input to the feature group function
            cpu_size (str): Size of the cpu for the feature group function
            memory (int): Memory (in GB) for the feature group function
        """
        function_source = inspect.getsource(function)
        return self.create_feature_group_from_function(function_source_code=function_source, function_name=function.__name__, table_name=table_name, input_feature_groups=input_tables, cpu_size=cpu_size, memory=memory)

    def create_algorithm_from_function(self, name: str, problem_type: str, training_data_parameter_names_mapping: dict = None, training_config_parameter_name: str = None, train_function: callable = None, predict_function: callable = None, predict_many_function: callable = None, initialize_function: callable = None, config_options: dict = None, is_default_enabled: bool = False, project_id: str = None):
        """
        Create a new algorithm, or update existing algorithm if the name already exists

        Args:
            name (String): The name to identify the algorithm, only uppercase letters, numbers and underscore allowed
            problem_type (Enum string): The type of the problem this algorithm will work on
            train_function (callable): The training fucntion callable to serialize and upload
            predict_function (callable): The predict function callable to serialize and upload
            predict_many_function (callable): The predict many function callable to serialize and upload
            initialize_function (callable): The initialize function callable to serialize and upload
            training_data_parameter_names_mapping (Dict): The mapping from feature group types to training data parameter names in the train function
            training_config_parameter_name (string): The train config parameter name in the train function
            config_options (Dict): Map dataset types and configs to train function parameter names
            is_default_enabled: Whether train with the algorithm by default
            project_id (Unique String Identifier): The unique version ID of the project
        """
        source_code, train_function_name, predict_function_name, predict_many_function_name, initialize_function_name = get_source_code_info(
            train_function, predict_function, predict_many_function, initialize_function)
        return self.create_algorithm(
            name=name,
            problem_type=problem_type,
            source_code=source_code,
            training_data_parameter_names_mapping=training_data_parameter_names_mapping,
            training_config_parameter_name=training_config_parameter_name,
            train_function_name=train_function_name,
            predict_function_name=predict_function_name,
            predict_many_function_name=predict_many_function_name,
            initialize_function_name=initialize_function_name,
            config_options=config_options,
            is_default_enabled=is_default_enabled,
            project_id=project_id)

    def update_algorithm_from_function(self, algorithm: str, training_data_parameter_names_mapping: dict = None, training_config_parameter_name: str = None, train_function: callable = None, predict_function: callable = None, predict_many_function: callable = None, initialize_function: callable = None, config_options: dict = None, is_default_enabled: bool = None):
        """
        Create a new algorithm, or update existing algorithm if the name already exists

        Args:
            algorithm (String): The name to identify the algorithm, only uppercase letters, numbers and underscore allowed
            train_function (callable): The training fucntion callable to serialize and upload
            predict_function (callable): The predict function callable to serialize and upload
            predict_many_function (callable): The predict many function callable to serialize and upload
            initialize_function (callable): The initialize function callable to serialize and upload
            training_data_parameter_names_mapping (Dict): The mapping from feature group types to training data parameter names in the train function
            training_config_parameter_name (string): The train config parameter name in the train function
            config_options (Dict): Map dataset types and configs to train function parameter names
            is_default_enabled: Whether train with the algorithm by default
        """
        source_code, train_function_name, predict_function_name, predict_many_function_name, initialize_function_name = get_source_code_info(
            train_function, predict_function, predict_many_function, initialize_function)
        return self.update_algorithm(
            algorithm=algorithm,
            source_code=source_code,
            training_data_parameter_names_mapping=training_data_parameter_names_mapping,
            training_config_parameter_name=training_config_parameter_name,
            train_function_name=train_function_name,
            predict_function_name=predict_function_name,
            predict_many_function_name=predict_many_function_name,
            initialize_function_name=initialize_function_name,
            config_options=config_options,
            is_default_enabled=is_default_enabled)

    def get_train_function_input(self, project_id: str, training_table_names: list = None, training_data_parameter_name_override: dict = None, training_config_parameter_name_override: str = None, training_config: dict = None, custom_algorithm_config: any = None):
        """
        Get the input data for the train function to test locally.

        Args:
            project_id (String): The id of the project
            training_table_names (List): A list of feature group tables used for training
            training_data_parameter_name_override (Dict): The mapping from feature group types to training data parameter names in the train function
            training_config_parameter_name_override (String): The train config parameter name in the train function
            training_config (Dict): A dictionary for Abacus.AI defined training options and values
            custom_algorithm_config (Any): User-defined config that can be serialized by JSON

        Returns:
            A dictionary that maps train function parameter names to their values.
        """

        train_function_info = self.get_custom_train_function_info(project_id=project_id, feature_group_names_for_training=training_table_names,
                                                                  training_data_parameter_name_override=training_data_parameter_name_override, training_config=training_config, custom_algorithm_config=custom_algorithm_config)
        train_data_parameter_to_feature_group_ids = train_function_info.train_data_parameter_to_feature_group_ids

        input = {parameter_name: self.describe_feature_group(fgid).load_as_pandas(
        ) for parameter_name, fgid in train_data_parameter_to_feature_group_ids.items()}
        input['schema_mappings'] = train_function_info.schema_mappings
        training_config_parameter_name = training_config_parameter_name_override or 'training_config'
        input[training_config_parameter_name] = train_function_info.training_config
        return input

    def train_model_with_algorithms(self, project_id: str, model_name: str, user_defined_algorithms: list, training_table_names: list, cpu_size: str = 'SMALL', memory: int = 3, user_defined_algorithms_only: bool = False, training_config: dict = None, user_defined_algorithm_configs: dict = None):
        """
        Train a model with provided user-defined algorithms.

        Args:
            project_id (String): The id of the project
            model_name (String): The name of the model to train
            user_defined_algorithms (List): A list of user-defined algorithm names
            training_table_names (List): A list of feature group tables used for training
            cpu_size (Enum): How much cpu is needed for the user-defined algorithms during training
            memory (Int): How much memory in GB is needed for the user-defined algorithms during training
            user_defined_algorithms_only (Boolean): Whether only train with user-defined algorithms, or also include Abacus.AI algorithms
            training_config (Dict): A dictionary for Abacus.AI defined training options and values
            user_defined_algorithm_configs (Dict): Configs for each user-defined algorithm, key is algorithm name, value is the config serialized to json
        """
        feature_group_ids = [self.describe_feature_group_by_table_name(
            table_name).feature_group_id for table_name in training_table_names]
        return self.train_model(project_id=project_id, name=model_name, training_config=training_config, feature_group_ids=feature_group_ids, custom_algorithms=user_defined_algorithms, custom_algorithms_only=user_defined_algorithms_only, custom_algorithm_configs=user_defined_algorithm_configs, cpu_size=cpu_size, memory=memory)

    def add_user_to_organization(self, email: str):
        """Invites a user to your organization. This method will send the specified email address an invitation link to join your organization.

        Args:
            email (str): The email address to invite to your Organization."""
        return self._call_api('addUserToOrganization', 'POST', query_params={}, body={'email': email})

    def create_organization_group(self, group_name: str, permissions: list, default_group: bool = False) -> OrganizationGroup:
        """Creates a new Organization Group.

        Args:
            group_name (str): The name of the group
            permissions (list): The list of permissions to initialize the group with
            default_group (bool): If true, this group will replace the current default group

        Returns:
            OrganizationGroup: Information about the created Organization Group"""
        return self._call_api('createOrganizationGroup', 'POST', query_params={}, body={'groupName': group_name, 'permissions': permissions, 'defaultGroup': default_group}, parse_type=OrganizationGroup)

    def add_organization_group_permission(self, organization_group_id: str, permission: str):
        """Adds a permission to the specified Organization Group

        Args:
            organization_group_id (str): The ID of the Organization Group
            permission (str): The permission to add to the Organization Group"""
        return self._call_api('addOrganizationGroupPermission', 'POST', query_params={}, body={'organizationGroupId': organization_group_id, 'permission': permission})

    def remove_organization_group_permission(self, organization_group_id: str, permission: str):
        """Removes a permission from the specified Organization Group

        Args:
            organization_group_id (str): The ID of the Organization Group
            permission (str): The permission to remove from the Organization Group"""
        return self._call_api('removeOrganizationGroupPermission', 'POST', query_params={}, body={'organizationGroupId': organization_group_id, 'permission': permission})

    def delete_organization_group(self, organization_group_id: str):
        """Deletes the specified Organization Group from the organization.

        Args:
            organization_group_id (str): The ID of the Organization Group"""
        return self._call_api('deleteOrganizationGroup', 'DELETE', query_params={'organizationGroupId': organization_group_id})

    def add_user_to_organization_group(self, organization_group_id: str, email: str):
        """Adds a user to the specified Organization Group

        Args:
            organization_group_id (str): The ID of the Organization Group
            email (str): The email of the user that is added to the group"""
        return self._call_api('addUserToOrganizationGroup', 'POST', query_params={}, body={'organizationGroupId': organization_group_id, 'email': email})

    def remove_user_from_organization_group(self, organization_group_id: str, email: str):
        """Removes a user from an Organization Group

        Args:
            organization_group_id (str): The ID of the Organization Group
            email (str): The email of the user to remove"""
        return self._call_api('removeUserFromOrganizationGroup', 'DELETE', query_params={'organizationGroupId': organization_group_id, 'email': email})

    def set_default_organization_group(self, organization_group_id: str):
        """Sets the default Organization Group that all new users that join an organization are automatically added to

        Args:
            organization_group_id (str): The ID of the Organization Group"""
        return self._call_api('setDefaultOrganizationGroup', 'POST', query_params={}, body={'organizationGroupId': organization_group_id})

    def delete_api_key(self, api_key_id: str):
        """Delete a specified API Key. You can use the "listApiKeys" method to find the list of all API Key IDs.

        Args:
            api_key_id (str): The ID of the API key to delete."""
        return self._call_api('deleteApiKey', 'DELETE', query_params={'apiKeyId': api_key_id})

    def remove_user_from_organization(self, email: str):
        """Removes the specified user from the Organization. You can remove yourself, Otherwise you must be an Organization Administrator to use this method to remove other users from the organization.

        Args:
            email (str): The email address of the user to remove from the Organization."""
        return self._call_api('removeUserFromOrganization', 'DELETE', query_params={'email': email})

    def create_project(self, name: str, use_case: str) -> Project:
        """Creates a project with your specified project name and use case. Creating a project creates a container for all of the datasets and the models that are associated with a particular problem/project that you would like to work on. For example, if you want to create a model to detect fraud, you have to first create a project, upload datasets, create feature groups, and then create one or more models to get predictions for your use case.

        Args:
            name (str): The project's name
            use_case (str): The use case that the project solves. You can refer to our (guide on use cases)[https://api.abacus.ai/app/help/useCases] for further details of each use case. The following enums are currently available for you to choose from:  LANGUAGE_DETECTION,  NLP_SENTIMENT,  NLP_QA,  NLP_SEARCH,  NLP_SENTENCE_BOUNDARY_DETECTION,  NLP_CLASSIFICATION,  NLP_SUMMARIZATION,  NLP_DOCUMENT_VISUALIZATION,  EMBEDDINGS_ONLY,  MODEL_WITH_EMBEDDINGS,  TORCH_MODEL_WITH_EMBEDDINGS,  PYTHON_MODEL,  NOTEBOOK_PYTHON_MODEL,  DOCKER_MODEL,  DOCKER_MODEL_WITH_EMBEDDINGS,  CUSTOMER_CHURN,  ENERGY,  FINANCIAL_METRICS,  CUMULATIVE_FORECASTING,  FRAUD_ACCOUNT,  FRAUD_THREAT,  FRAUD_TRANSACTIONS,  OPERATIONS_CLOUD,  CLOUD_SPEND,  TIMESERIES_ANOMALY_DETECTION,  OPERATIONS_MAINTENANCE,  OPERATIONS_INCIDENT,  PERS_PROMOTIONS,  PREDICTING,  FEATURE_STORE,  RETAIL,  SALES_FORECASTING,  SALES_SCORING,  FEED_RECOMMEND,  USER_RANKINGS,  NAMED_ENTITY_RECOGNITION,  USER_ITEM_AFFINITY,  USER_RECOMMENDATIONS,  USER_RELATED,  VISION_SEGMENTATION,  VISION,  FEATURE_DRIFT,  SCHEDULING,  GENERIC_FORECASTING.

        Returns:
            Project: This object represents the newly created project. For details refer to"""
        return self._call_api('createProject', 'POST', query_params={}, body={'name': name, 'useCase': use_case}, parse_type=Project)

    def rename_project(self, project_id: str, name: str):
        """This method renames a project after it is created.

        Args:
            project_id (str): The unique ID for the project.
            name (str): The new name for the project."""
        return self._call_api('renameProject', 'PATCH', query_params={}, body={'projectId': project_id, 'name': name})

    def delete_project(self, project_id: str):
        """Deletes a specified project from your organization.

        This method deletes the project, trained models and deployments in the specified project. The datasets attached to the specified project remain available for use with other projects in the organization.

        This method will not delete a project that contains active deployments. Be sure to stop all active deployments before you use the delete option.

        Note: All projects, models, and deployments cannot be recovered once they are deleted.


        Args:
            project_id (str): The unique ID of the project to delete."""
        return self._call_api('deleteProject', 'DELETE', query_params={'projectId': project_id})

    def add_feature_group_to_project(self, feature_group_id: str, project_id: str, feature_group_type: str = 'CUSTOM_TABLE', feature_group_use: str = None):
        """Adds a feature group to a project,

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            project_id (str): The unique ID associated with the project.
            feature_group_type (str):  The feature group type of the feature group. The type is based on the use case under which the feature group is being created. For example, Catalog Attributes can be a feature group type under personalized recommendation use case.
            feature_group_use (str): The user assigned feature group use which allows for organizing project feature groups  DATA_WRANGLING,  TRAINING_INPUT,  BATCH_PREDICTION_INPUT"""
        return self._call_api('addFeatureGroupToProject', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'projectId': project_id, 'featureGroupType': feature_group_type, 'featureGroupUse': feature_group_use})

    def remove_feature_group_from_project(self, feature_group_id: str, project_id: str):
        """Removes a feature group from a project.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            project_id (str): The unique ID associated with the project."""
        return self._call_api('removeFeatureGroupFromProject', 'DELETE', query_params={'featureGroupId': feature_group_id, 'projectId': project_id})

    def set_feature_group_type(self, feature_group_id: str, project_id: str, feature_group_type: str = 'CUSTOM_TABLE'):
        """Update the feature group type in a project. The feature group must already be added to the project.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            project_id (str): The unique ID associated with the project.
            feature_group_type (str): The feature group type to set the feature group as. The type is based on the use case under which the feature group is being created. For example, Catalog Attributes can be a feature group type under personalized recommendation use case."""
        return self._call_api('setFeatureGroupType', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'projectId': project_id, 'featureGroupType': feature_group_type})

    def use_feature_group_for_training(self, feature_group_id: str, project_id: str, use_for_training: bool = True):
        """Use the feature group for model training input

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            project_id (str): The unique ID associated with the project.
            use_for_training (bool): Boolean variable to include or exclude a feature group from a model's training. Only one feature group per type can be used for training"""
        return self._call_api('useFeatureGroupForTraining', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'projectId': project_id, 'useForTraining': use_for_training})

    def set_feature_mapping(self, project_id: str, feature_group_id: str, feature_name: str, feature_mapping: str, nested_column_name: str = None) -> List[Feature]:
        """Set a column's feature mapping. If the column mapping is single-use and already set in another column in this feature group, this call will first remove the other column's mapping and move it to this column.

        Args:
            project_id (str): The unique ID associated with the project.
            feature_group_id (str): The unique ID associated with the feature group.
            feature_name (str): The name of the feature.
            feature_mapping (str): The mapping of the feature in the feature group.
            nested_column_name (str): The name of the nested column.

        Returns:
            Feature: A list of objects that describes the resulting feature group's schema after the feature's featureMapping is set."""
        return self._call_api('setFeatureMapping', 'POST', query_params={}, body={'projectId': project_id, 'featureGroupId': feature_group_id, 'featureName': feature_name, 'featureMapping': feature_mapping, 'nestedColumnName': nested_column_name}, parse_type=Feature)

    def set_column_data_type(self, project_id: str, dataset_id: str, column: str, data_type: str) -> List[Schema]:
        """Set a dataset's column type.

        Args:
            project_id (str): The unique ID associated with the project.
            dataset_id (str): The unique ID associated with the dataset.
            column (str): The name of the column.
            data_type (str): The type of the data in the column.  CATEGORICAL,  CATEGORICAL_LIST,  NUMERICAL,  TIMESTAMP,  TEXT,  EMAIL,  LABEL_LIST,  JSON,  OBJECT_REFERENCE Refer to the (guide on feature types)[https://api.abacus.ai/app/help/class/FeatureType] for more information. Note: Some ColumnMappings will restrict the options or explicity set the DataType.

        Returns:
            Schema: A list of objects that describes the resulting dataset's schema after the column's dataType is set."""
        return self._call_api('setColumnDataType', 'POST', query_params={'datasetId': dataset_id}, body={'projectId': project_id, 'column': column, 'dataType': data_type}, parse_type=Schema)

    def set_column_mapping(self, project_id: str, dataset_id: str, column: str, column_mapping: str) -> List[Schema]:
        """Set a dataset's column mapping. If the column mapping is single-use and already set in another column in this dataset, this call will first remove the other column's mapping and move it to this column.

        Args:
            project_id (str): The unique ID associated with the project.
            dataset_id (str): The unique ID associated with the dataset.
            column (str): The name of the column.
            column_mapping (str): The mapping of the column in the dataset. See a list of columns mapping enums here.

        Returns:
            Schema: A list of columns that describes the resulting dataset's schema after the column's columnMapping is set."""
        return self._call_api('setColumnMapping', 'POST', query_params={'datasetId': dataset_id}, body={'projectId': project_id, 'column': column, 'columnMapping': column_mapping}, parse_type=Schema)

    def remove_column_mapping(self, project_id: str, dataset_id: str, column: str) -> List[Schema]:
        """Removes a column mapping from a column in the dataset. Returns a list of all columns with their mappings once the change is made.

        Args:
            project_id (str): The unique ID associated with the project.
            dataset_id (str): The unique ID associated with the dataset.
            column (str): The name of the column.

        Returns:
            Schema: A list of objects that describes the resulting dataset's schema after the column's columnMapping is set."""
        return self._call_api('removeColumnMapping', 'DELETE', query_params={'projectId': project_id, 'datasetId': dataset_id, 'column': column}, parse_type=Schema)

    def create_feature_group(self, table_name: str, sql: str, description: str = None) -> FeatureGroup:
        """Creates a new feature group from a SQL statement.

        Args:
            table_name (str): The unique name to be given to the feature group.
            sql (str): Input SQL statement for forming the feature group.
            description (str): The description about the feature group.

        Returns:
            FeatureGroup: The created feature group"""
        return self._call_api('createFeatureGroup', 'POST', query_params={}, body={'tableName': table_name, 'sql': sql, 'description': description}, parse_type=FeatureGroup)

    def create_feature_group_from_template(self, table_name: str, feature_group_template_id: str, template_bindings: list = None, should_attach_feature_group_to_template: bool = True, description: str = None) -> FeatureGroup:
        """Creates a new feature group from a SQL statement.

        Args:
            table_name (str): The unique name to be given to the feature group.
            feature_group_template_id (str): template_info.template_sqlThe unique ID associated with the template that will be used to create this feature group.
            template_bindings (list): Variable bindings that override the template's variable values.
            should_attach_feature_group_to_template (bool): Set to False to create a feature group but not leave it attached the template that created it.
            description (str): A user-friendly description of this feature group.

        Returns:
            FeatureGroup: The created feature group"""
        return self._call_api('createFeatureGroupFromTemplate', 'POST', query_params={}, body={'tableName': table_name, 'featureGroupTemplateId': feature_group_template_id, 'templateBindings': template_bindings, 'shouldAttachFeatureGroupToTemplate': should_attach_feature_group_to_template, 'description': description}, parse_type=FeatureGroup)

    def create_feature_group_from_function(self, table_name: str, function_source_code: str, function_name: str, input_feature_groups: list = [], description: str = None, cpu_size: str = None, memory: int = None, package_requirements: dict = None) -> FeatureGroup:
        """Creates a new feature in a Feature Group from user provided code. Code language currently supported is Python.

        If a list of input feature groups are supplied, we will provide as arguments to the function DataFrame's
        (pandas in the case of Python) with the materialized feature groups for those input feature groups.

        This method expects `function_source_code to be a valid language source file which contains a function named
        `function_name`. This function needs return a DataFrame when it is executed and this DataFrame will be used
        as the materialized version of this feature group table.


        Args:
            table_name (str): The unique name to be given to the feature group.
            function_source_code (str): Contents of a valid source code file in a supported Feature Group specification language (currently only Python). The source code should contain a function called function_name. A list of allowed import and system libraries for each language is specified in the user functions documentation section.
            function_name (str): Name of the function found in the source code that will be executed (on the optional inputs) to materialize this feature group.
            input_feature_groups (list): List of feature groups that are supplied to the function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            description (str): The description for this feature group.
            cpu_size (str): Size of the cpu for the feature group function
            memory (int): Memory (in GB) for the feature group function
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency

        Returns:
            FeatureGroup: The created feature group"""
        return self._call_api('createFeatureGroupFromFunction', 'POST', query_params={}, body={'tableName': table_name, 'functionSourceCode': function_source_code, 'functionName': function_name, 'inputFeatureGroups': input_feature_groups, 'description': description, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements}, parse_type=FeatureGroup)

    def create_feature_group_from_zip(self, table_name: str, function_name: str, module_name: str, input_feature_groups: list = None, description: str = None, cpu_size: str = None, memory: int = None, package_requirements: dict = None) -> Upload:
        """Creates a new feature group from a ZIP file.

        Args:
            table_name (str): The unique name to be given to the feature group.
            function_name (str): Name of the function found in the module that will be executed (on the optional inputs) to materialize this feature group.
            module_name (str): Path to the file with the feature group function.
            input_feature_groups (list): List of feature groups that are supplied to the function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            description (str): The description about the feature group.
            cpu_size (str): Size of the cpu for the feature group function
            memory (int): Memory (in GB) for the feature group function
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency

        Returns:
            Upload: The Upload to upload the zip file to"""
        return self._call_api('createFeatureGroupFromZip', 'POST', query_params={}, body={'tableName': table_name, 'functionName': function_name, 'moduleName': module_name, 'inputFeatureGroups': input_feature_groups, 'description': description, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements}, parse_type=Upload)

    def create_feature_group_from_git(self, application_connector_id: str, branch_name: str, table_name: str, function_name: str, module_name: str, python_root: str = None, input_feature_groups: list = None, description: str = None, cpu_size: str = None, memory: int = None, package_requirements: dict = None) -> FeatureGroup:
        """Creates a new feature group from a ZIP file.

        Args:
            application_connector_id (str): The unique ID associated with the git application connector.
            branch_name (str): Name of the branch in the git repository to be used for training.
            table_name (str): The unique name to be given to the feature group.
            function_name (str): Name of the function found in the module that will be executed (on the optional inputs) to materialize this feature group.
            module_name (str): Path to the file with the feature group function.
            python_root (str): Path from the top level of the git repository to the directory containing the Python source code. If not provided, the default is the root of the git repository.
            input_feature_groups (list): List of feature groups that are supplied to the function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            description (str): The description about the feature group.
            cpu_size (str): Size of the cpu for the feature group function
            memory (int): Memory (in GB) for the feature group function
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency

        Returns:
            FeatureGroup: The created feature group"""
        return self._call_api('createFeatureGroupFromGit', 'POST', query_params={}, body={'applicationConnectorId': application_connector_id, 'branchName': branch_name, 'tableName': table_name, 'functionName': function_name, 'moduleName': module_name, 'pythonRoot': python_root, 'inputFeatureGroups': input_feature_groups, 'description': description, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements}, parse_type=FeatureGroup)

    def create_sampling_feature_group(self, feature_group_id: str, table_name: str, sampling_config: dict, description: str = None) -> FeatureGroup:
        """Creates a new feature group defined as a sample of rows from another feature group.

        For efficiency, sampling is approximate unless otherwise specified. (E.g. the number of rows may vary slightly from what was requested).


        Args:
            feature_group_id (str): The unique ID associated with the pre-existing feature group that will be sampled by this new feature group. I.e. the input for sampling.
            table_name (str): The unique name to be given to this sampling feature group.
            sampling_config (dict): JSON object (aka map) defining the sampling method and its parameters.
            description (str): A human-readable description of this feature group.

        Returns:
            FeatureGroup: The created feature group."""
        return self._call_api('createSamplingFeatureGroup', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'tableName': table_name, 'samplingConfig': sampling_config, 'description': description}, parse_type=FeatureGroup)

    def create_merge_feature_group(self, source_feature_group_id: str, table_name: str, merge_config: dict, description: str = None) -> FeatureGroup:
        """Creates a new feature group defined as the union of other feature group versions.

        Args:
            source_feature_group_id (str): ID corresponding to the dataset feature group that will have its versions merged into this feature group.
            table_name (str): The unique name to be given to this merge feature group.
            merge_config (dict): JSON object (aka map) defining the merging method and its parameters.
            description (str): A human-readable description of this feature group.

        Returns:
            FeatureGroup: The created feature group."""
        return self._call_api('createMergeFeatureGroup', 'POST', query_params={}, body={'sourceFeatureGroupId': source_feature_group_id, 'tableName': table_name, 'mergeConfig': merge_config, 'description': description}, parse_type=FeatureGroup)

    def create_transform_feature_group(self, source_feature_group_id: str, table_name: str, transform_config: dict, description: str = None) -> FeatureGroup:
        """Creates a new feature group defined as a pre-defined transform on another feature group.

        Args:
            source_feature_group_id (str): ID corresponding to the feature group that will have the transformation applied.
            table_name (str): The unique name to be given to this transform feature group.
            transform_config (dict): JSON object (aka map) defining the transform and its parameters.
            description (str): A human-readable description of this feature group.

        Returns:
            FeatureGroup: The created feature group."""
        return self._call_api('createTransformFeatureGroup', 'POST', query_params={}, body={'sourceFeatureGroupId': source_feature_group_id, 'tableName': table_name, 'transformConfig': transform_config, 'description': description}, parse_type=FeatureGroup)

    def create_snapshot_feature_group(self, feature_group_version: str, table_name: str) -> FeatureGroup:
        """Creates a Snapshot Feature Group corresponding to a specific feature group version.

        Args:
            feature_group_version (str): The unique ID associated with the feature group version being snapshotted.
            table_name (str): The name for the newly created Snapshot Feature Group table.

        Returns:
            FeatureGroup: Feature Group corresponding to the newly created Snapshot."""
        return self._call_api('createSnapshotFeatureGroup', 'POST', query_params={}, body={'featureGroupVersion': feature_group_version, 'tableName': table_name}, parse_type=FeatureGroup)

    def set_feature_group_sampling_config(self, feature_group_id: str, sampling_config: dict) -> FeatureGroup:
        """Set a FeatureGroup’s sampling to the config values provided, so that the rows the FeatureGroup returns will be a sample of those it would otherwise have returned.

        Currently, sampling is only for Sampling FeatureGroups, so this API only allows calling on that kind of FeatureGroup.


        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            sampling_config (dict): A json object string specifying the sampling method and parameters specific to that sampling method. Empty sampling_config means no sampling.

        Returns:
            FeatureGroup: The updated feature group."""
        return self._call_api('setFeatureGroupSamplingConfig', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'samplingConfig': sampling_config}, parse_type=FeatureGroup)

    def set_feature_group_merge_config(self, feature_group_id: str, merge_config: dict) -> None:
        """Set a MergeFeatureGroup’s merge config to the values provided, so that the feature group only returns a bounded range of an incremental dataset.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            merge_config (dict): A json object string specifying the merge rule. An empty mergeConfig will default to only including the latest Dataset Version."""
        return self._call_api('setFeatureGroupMergeConfig', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'mergeConfig': merge_config})

    def set_feature_group_transform_config(self, feature_group_id: str, transform_config: dict) -> None:
        """Set a TransformFeatureGroup’s transform config to the values provided.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            transform_config (dict): A json object string specifying the pre-defined transformation."""
        return self._call_api('setFeatureGroupTransformConfig', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'transformConfig': transform_config})

    def set_feature_group_schema(self, feature_group_id: str, schema: list):
        """Creates a new schema and points the feature group to the new feature group schema id.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            schema (list): An array of json objects with 'name' and 'dataType' properties."""
        return self._call_api('setFeatureGroupSchema', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'schema': schema})

    def create_feature(self, feature_group_id: str, name: str, select_expression: str) -> FeatureGroup:
        """Creates a new feature in a Feature Group from a SQL select statement

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            name (str): The name of the feature to add
            select_expression (str): SQL select expression to create the feature

        Returns:
            FeatureGroup: A feature group object with the newly added feature."""
        return self._call_api('createFeature', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'name': name, 'selectExpression': select_expression}, parse_type=FeatureGroup)

    def add_feature_group_tag(self, feature_group_id: str, tag: str):
        """Adds a tag to the feature group

        Args:
            feature_group_id (str): The feature group
            tag (str): The tag to add to the feature group"""
        return self._call_api('addFeatureGroupTag', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'tag': tag})

    def remove_feature_group_tag(self, feature_group_id: str, tag: str):
        """Removes a tag from the feature group

        Args:
            feature_group_id (str): The feature group
            tag (str): The tag to add to the feature group"""
        return self._call_api('removeFeatureGroupTag', 'DELETE', query_params={'featureGroupId': feature_group_id, 'tag': tag})

    def add_feature_tag(self, feature_group_id: str, feature: str, tag: str):
        """

        Args:
            feature_group_id (str): 
            feature (str): 
            tag (str): """
        return self._call_api('addFeatureTag', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'feature': feature, 'tag': tag})

    def remove_feature_tag(self, feature_group_id: str, feature: str, tag: str):
        """

        Args:
            feature_group_id (str): 
            feature (str): 
            tag (str): """
        return self._call_api('removeFeatureTag', 'DELETE', query_params={'featureGroupId': feature_group_id, 'feature': feature, 'tag': tag})

    def create_nested_feature(self, feature_group_id: str, nested_feature_name: str, table_name: str, using_clause: str, where_clause: str = None, order_clause: str = None) -> FeatureGroup:
        """Creates a new nested feature in a feature group from a SQL statements to create a new nested feature.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            nested_feature_name (str): The name of the feature.
            table_name (str): The table name of the feature group to nest
            using_clause (str): The SQL join column or logic to join the nested table with the parent
            where_clause (str): A SQL where statement to filter the nested rows
            order_clause (str): A SQL clause to order the nested rows

        Returns:
            FeatureGroup: A feature group object with the newly added nested feature."""
        return self._call_api('createNestedFeature', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'nestedFeatureName': nested_feature_name, 'tableName': table_name, 'usingClause': using_clause, 'whereClause': where_clause, 'orderClause': order_clause}, parse_type=FeatureGroup)

    def update_nested_feature(self, feature_group_id: str, nested_feature_name: str, table_name: str = None, using_clause: str = None, where_clause: str = None, order_clause: str = None, new_nested_feature_name: str = None) -> FeatureGroup:
        """Updates a previously existing nested feature in a feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            nested_feature_name (str): The name of the feature to be updated.
            table_name (str): The name of the table.
            using_clause (str): The SQL join column or logic to join the nested table with the parent
            where_clause (str): A SQL where statement to filter the nested rows
            order_clause (str): A SQL clause to order the nested rows
            new_nested_feature_name (str): New name for the nested feature.

        Returns:
            FeatureGroup: A feature group object with the updated nested feature."""
        return self._call_api('updateNestedFeature', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'nestedFeatureName': nested_feature_name, 'tableName': table_name, 'usingClause': using_clause, 'whereClause': where_clause, 'orderClause': order_clause, 'newNestedFeatureName': new_nested_feature_name}, parse_type=FeatureGroup)

    def delete_nested_feature(self, feature_group_id: str, nested_feature_name: str) -> FeatureGroup:
        """Delete a nested feature.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            nested_feature_name (str): The name of the feature to be updated.

        Returns:
            FeatureGroup: A feature group object without the deleted nested feature."""
        return self._call_api('deleteNestedFeature', 'DELETE', query_params={'featureGroupId': feature_group_id, 'nestedFeatureName': nested_feature_name}, parse_type=FeatureGroup)

    def create_point_in_time_feature(self, feature_group_id: str, feature_name: str, history_table_name: str, aggregation_keys: list, timestamp_key: str, historical_timestamp_key: str, expression: str, lookback_window_seconds: float = None, lookback_window_lag_seconds: float = 0, lookback_count: int = None, lookback_until_position: int = 0) -> FeatureGroup:
        """Creates a new point in time feature in a feature group using another historical feature group, window spec and aggregate expression.

        We use the aggregation keys, and either the lookbackWindowSeconds or the lookbackCount values to perform the window aggregation for every row in the current feature group.
        If the window is specified in seconds, then all rows in the history table which match the aggregation keys and with historicalTimeFeature >= lookbackStartCount and < the value
        of the current rows timeFeature are considered. An option lookbackWindowLagSeconds (+ve or -ve) can be used to offset the current value of the timeFeature. If this value
        is negative, we will look at the future rows in the history table, so care must be taken to make sure that these rows are available in the online context when we are performing
        a lookup on this feature group. If window is specified in counts, then we order the historical table rows aligning by time and consider rows from the window where
        the rank order is >= lookbackCount and includes the row just prior to the current one. The lag is specified in term of positions using lookbackUntilPosition.


        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            feature_name (str): The name of the feature to create
            history_table_name (str): The table name of the history table.
            aggregation_keys (list): List of keys to use for join the historical table and performing the window aggregation.
            timestamp_key (str): Name of feature which contains the timestamp value for the point in time feature
            historical_timestamp_key (str): Name of feature which contains the historical timestamp.
            expression (str): SQL Aggregate expression which can convert a sequence of rows into a scalar value.
            lookback_window_seconds (float): If window is specified in terms of time, number of seconds in the past from the current time for start of the window.
            lookback_window_lag_seconds (float): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window. If it is negative, we are looking at the "future" rows in the history table.
            lookback_count (int): If window is specified in terms of count, the start position of the window (0 is the current row)
            lookback_until_position (int): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window by that many rows. If it is negative, we are looking at those many "future" rows in the history table.

        Returns:
            FeatureGroup: A feature group object with the newly added nested feature."""
        return self._call_api('createPointInTimeFeature', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'featureName': feature_name, 'historyTableName': history_table_name, 'aggregationKeys': aggregation_keys, 'timestampKey': timestamp_key, 'historicalTimestampKey': historical_timestamp_key, 'expression': expression, 'lookbackWindowSeconds': lookback_window_seconds, 'lookbackWindowLagSeconds': lookback_window_lag_seconds, 'lookbackCount': lookback_count, 'lookbackUntilPosition': lookback_until_position}, parse_type=FeatureGroup)

    def update_point_in_time_feature(self, feature_group_id: str, feature_name: str, history_table_name: str = None, aggregation_keys: list = None, timestamp_key: str = None, historical_timestamp_key: str = None, expression: str = None, lookback_window_seconds: float = None, lookback_window_lag_seconds: float = None, lookback_count: int = None, lookback_until_position: int = None, new_feature_name: str = None) -> FeatureGroup:
        """Updates an existing point in time feature in a feature group. See createPointInTimeFeature for detailed semantics.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            feature_name (str): The name of the feature.
            history_table_name (str): The table name of the history table. If not specified, we use the current table to do a self join.
            aggregation_keys (list): List of keys to use for join the historical table and performing the window aggregation.
            timestamp_key (str): Name of feature which contains the timestamp value for the point in time feature
            historical_timestamp_key (str): Name of feature which contains the historical timestamp.
            expression (str): SQL Aggregate expression which can convert a sequence of rows into a scalar value.
            lookback_window_seconds (float): If window is specified in terms of time, number of seconds in the past from the current time for start of the window.
            lookback_window_lag_seconds (float): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window. If it is negative, we are looking at the "future" rows in the history table.
            lookback_count (int): If window is specified in terms of count, the start position of the window (0 is the current row)
            lookback_until_position (int): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window by that many rows. If it is negative, we are looking at those many "future" rows in the history table.
            new_feature_name (str): New name for the point in time feature.

        Returns:
            FeatureGroup: A feature group object with the newly added nested feature."""
        return self._call_api('updatePointInTimeFeature', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'featureName': feature_name, 'historyTableName': history_table_name, 'aggregationKeys': aggregation_keys, 'timestampKey': timestamp_key, 'historicalTimestampKey': historical_timestamp_key, 'expression': expression, 'lookbackWindowSeconds': lookback_window_seconds, 'lookbackWindowLagSeconds': lookback_window_lag_seconds, 'lookbackCount': lookback_count, 'lookbackUntilPosition': lookback_until_position, 'newFeatureName': new_feature_name}, parse_type=FeatureGroup)

    def create_point_in_time_group(self, feature_group_id: str, group_name: str, window_key: str, aggregation_keys: list, history_table_name: str = None, history_window_key: str = None, history_aggregation_keys: list = None, lookback_window: float = None, lookback_window_lag: float = 0, lookback_count: int = None, lookback_until_position: int = 0) -> FeatureGroup:
        """Create point in time group

        Args:
            feature_group_id (str): The unique ID associated with the feature group to add the point in time group to.
            group_name (str): The name of the point in time group
            window_key (str): Name of feature to use for ordering the rows on the source table
            aggregation_keys (list): List of keys to perform on the source table for the window aggregation.
            history_table_name (str): The table to use for aggregating, if not provided, the source table will be used
            history_window_key (str): Name of feature to use for ordering the rows on the history table. If not provided, the windowKey from the source table will be used
            history_aggregation_keys (list): List of keys to use for join the historical table and performing the window aggregation. If not provided, the aggregationKeys from the source table will be used. Must be the same length and order as the source table's aggregationKeys
            lookback_window (float): Number of seconds in the past from the current time for start of the window. If 0, the lookback will include all rows.
            lookback_window_lag (float): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window. If it is negative, we are looking at the "future" rows in the history table.
            lookback_count (int): If window is specified in terms of count, the start position of the window (0 is the current row)
            lookback_until_position (int): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window by that many rows. If it is negative, we are looking at those many "future" rows in the history table.

        Returns:
            FeatureGroup: The feature group after the point in time group has been created"""
        return self._call_api('createPointInTimeGroup', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'groupName': group_name, 'windowKey': window_key, 'aggregationKeys': aggregation_keys, 'historyTableName': history_table_name, 'historyWindowKey': history_window_key, 'historyAggregationKeys': history_aggregation_keys, 'lookbackWindow': lookback_window, 'lookbackWindowLag': lookback_window_lag, 'lookbackCount': lookback_count, 'lookbackUntilPosition': lookback_until_position}, parse_type=FeatureGroup)

    def update_point_in_time_group(self, feature_group_id: str, group_name: str, window_key: str = None, aggregation_keys: list = None, history_table_name: str = None, history_window_key: str = None, history_aggregation_keys: list = None, lookback_window: float = None, lookback_window_lag: float = None, lookback_count: int = None, lookback_until_position: int = None) -> FeatureGroup:
        """Update point in time group

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            group_name (str): The name of the point in time group
            window_key (str): Name of feature which contains the timestamp value for the point in time feature
            aggregation_keys (list): List of keys to use for join the historical table and performing the window aggregation.
            history_table_name (str): The table to use for aggregating, if not provided, the source table will be used
            history_window_key (str): Name of feature to use for ordering the rows on the history table. If not provided, the windowKey from the source table will be used
            history_aggregation_keys (list): List of keys to use for join the historical table and performing the window aggregation. If not provided, the aggregationKeys from the source table will be used. Must be the same length and order as the source table's aggregationKeys
            lookback_window (float): Number of seconds in the past from the current time for start of the window.
            lookback_window_lag (float): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window. If it is negative, we are looking at the "future" rows in the history table.
            lookback_count (int): If window is specified in terms of count, the start position of the window (0 is the current row)
            lookback_until_position (int): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window by that many rows. If it is negative, we are looking at those many "future" rows in the history table.

        Returns:
            FeatureGroup: The feature group after the update has been applied"""
        return self._call_api('updatePointInTimeGroup', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'groupName': group_name, 'windowKey': window_key, 'aggregationKeys': aggregation_keys, 'historyTableName': history_table_name, 'historyWindowKey': history_window_key, 'historyAggregationKeys': history_aggregation_keys, 'lookbackWindow': lookback_window, 'lookbackWindowLag': lookback_window_lag, 'lookbackCount': lookback_count, 'lookbackUntilPosition': lookback_until_position}, parse_type=FeatureGroup)

    def delete_point_in_time_group(self, feature_group_id: str, group_name: str) -> FeatureGroup:
        """Delete point in time group

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            group_name (str): The name of the point in time group

        Returns:
            FeatureGroup: The feature group after the point in time group has been deleted"""
        return self._call_api('deletePointInTimeGroup', 'DELETE', query_params={'featureGroupId': feature_group_id, 'groupName': group_name}, parse_type=FeatureGroup)

    def create_point_in_time_group_feature(self, feature_group_id: str, group_name: str, name: str, expression: str) -> FeatureGroup:
        """Create point in time group feature

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            group_name (str): The name of the point in time group
            name (str): The name of the feature to add to the point in time group
            expression (str): SQL Aggregate expression which can convert a sequence of rows into a scalar value.

        Returns:
            FeatureGroup: The feature group after the update has been applied"""
        return self._call_api('createPointInTimeGroupFeature', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'groupName': group_name, 'name': name, 'expression': expression}, parse_type=FeatureGroup)

    def update_point_in_time_group_feature(self, feature_group_id: str, group_name: str, name: str, expression: str) -> FeatureGroup:
        """Update a feature's SQL expression in a point in time group

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            group_name (str): The name of the point in time group
            name (str): The name of the feature to add to the point in time group
            expression (str): SQL Aggregate expression which can convert a sequence of rows into a scalar value.

        Returns:
            FeatureGroup: The feature group after the update has been applied"""
        return self._call_api('updatePointInTimeGroupFeature', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'groupName': group_name, 'name': name, 'expression': expression}, parse_type=FeatureGroup)

    def set_feature_type(self, feature_group_id: str, feature: str, feature_type: str) -> Schema:
        """Set a feature's type in a feature group/. Specify the feature group ID, feature name and feature type, and the method will return the new column with the resulting changes reflected.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            feature (str): The name of the feature.
            feature_type (str): The machine learning type of the data in the feature.  CATEGORICAL,  CATEGORICAL_LIST,  NUMERICAL,  TIMESTAMP,  TEXT,  EMAIL,  LABEL_LIST,  JSON,  OBJECT_REFERENCE Refer to the (guide on feature types)[https://api.abacus.ai/app/help/class/FeatureType] for more information. Note: Some FeatureMappings will restrict the options or explicitly set the FeatureType.

        Returns:
            Schema: The feature group after the data_type is applied"""
        return self._call_api('setFeatureType', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'feature': feature, 'featureType': feature_type}, parse_type=Schema)

    def invalidate_streaming_feature_group_data(self, feature_group_id: str, invalid_before_timestamp: int):
        """Invalidates all streaming data with timestamp before invalidBeforeTimestamp

        Args:
            feature_group_id (str): The Streaming feature group to record data to
            invalid_before_timestamp (int): The unix timestamp, any data which has a timestamp before this time will be deleted"""
        return self._call_api('invalidateStreamingFeatureGroupData', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'invalidBeforeTimestamp': invalid_before_timestamp})

    def concatenate_feature_group_data(self, feature_group_id: str, source_feature_group_id: str, merge_type: str = 'UNION', replace_until_timestamp: int = None, skip_materialize: bool = False):
        """Concatenates data from one feature group to another. Feature groups can be merged if their schema's are compatible and they have the special updateTimestampKey column and if set, the primaryKey column. The second operand in the concatenate operation will be appended to the first operand (merge target).

        Args:
            feature_group_id (str): The destination feature group.
            source_feature_group_id (str): The feature group to concatenate with the destination feature group.
            merge_type (str): UNION or INTERSECTION
            replace_until_timestamp (int): The unix timestamp to specify the point till which we will replace data from the source feature group.
            skip_materialize (bool): If true, will not materialize the concatenated feature group"""
        return self._call_api('concatenateFeatureGroupData', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'sourceFeatureGroupId': source_feature_group_id, 'mergeType': merge_type, 'replaceUntilTimestamp': replace_until_timestamp, 'skipMaterialize': skip_materialize})

    def remove_concatenation_config(self, feature_group_id: str):
        """Removes the concatenation config on a destination feature group.

        Args:
            feature_group_id (str): Removes the concatenation configuration on a destination feature group"""
        return self._call_api('removeConcatenationConfig', 'DELETE', query_params={'featureGroupId': feature_group_id})

    def set_feature_group_indexing_config(self, feature_group_id: str, primary_key: str = None, update_timestamp_key: str = None, lookup_keys: list = None):
        """Sets various attributes of the feature group used for deployment lookups and streaming updates.

        Args:
            feature_group_id (str): The feature group
            primary_key (str): Name of feature which defines the primary key of the feature group.
            update_timestamp_key (str): Name of feature which defines the update timestamp of the feature group - used in concatenation and primary key deduplication.
            lookup_keys (list): List of feature names which can be used in the lookup api to restrict the computation to a set of dataset rows. These feature names have to correspond to underlying dataset columns."""
        return self._call_api('setFeatureGroupIndexingConfig', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'primaryKey': primary_key, 'updateTimestampKey': update_timestamp_key, 'lookupKeys': lookup_keys})

    def update_feature_group(self, feature_group_id: str, description: str = None) -> FeatureGroup:
        """Modifies an existing feature group

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            description (str): The description about the feature group.

        Returns:
            FeatureGroup: The updated feature group object."""
        return self._call_api('updateFeatureGroup', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'description': description}, parse_type=FeatureGroup)

    def detach_feature_group_from_template(self, feature_group_id: str) -> FeatureGroup:
        """Update a feature group to detach it from a template.

        Currently, this converts the feature group into a SQL feature group rather than a template feature group.


        Args:
            feature_group_id (str): The unique ID associated with the feature group.

        Returns:
            FeatureGroup: The updated feature group"""
        return self._call_api('detachFeatureGroupFromTemplate', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id}, parse_type=FeatureGroup)

    def update_feature_group_template_bindings(self, feature_group_id: str, template_bindings: list = None) -> FeatureGroup:
        """Update the feature group template bindings for a template feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            template_bindings (list): Values in these bindings override values set in the template.

        Returns:
            FeatureGroup: The updated feature group"""
        return self._call_api('updateFeatureGroupTemplateBindings', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'templateBindings': template_bindings}, parse_type=FeatureGroup)

    def update_feature_group_sql_definition(self, feature_group_id: str, sql: str) -> FeatureGroup:
        """Updates the SQL statement for a feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            sql (str): Input SQL statement for the feature group.

        Returns:
            FeatureGroup: The updated feature group"""
        return self._call_api('updateFeatureGroupSqlDefinition', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'sql': sql}, parse_type=FeatureGroup)

    def update_dataset_feature_group_feature_expression(self, feature_group_id: str, feature_expression: str) -> FeatureGroup:
        """Updates the SQL feature expression for a dataset feature group's custom features

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            feature_expression (str): Input SQL statement for the feature group.

        Returns:
            FeatureGroup: The updated feature group"""
        return self._call_api('updateDatasetFeatureGroupFeatureExpression', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'featureExpression': feature_expression}, parse_type=FeatureGroup)

    def update_feature_group_function_definition(self, feature_group_id: str, function_source_code: str = None, function_name: str = None, input_feature_groups: list = None, cpu_size: str = None, memory: int = None, package_requirements: dict = None) -> FeatureGroup:
        """Updates the function definition for a feature group created using createFeatureGroupFromFunction

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            function_source_code (str): Contents of a valid source code file in a supported Feature Group specification language (currently only Python). The source code should contain a function called function_name. A list of allowed import and system libraries for each language is specified in the user functions documentation section.
            function_name (str): Name of the function found in the source code that will be executed (on the optional inputs) to materialize this feature group.
            input_feature_groups (list): List of feature groups that are supplied to the function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            cpu_size (str): Size of the cpu for the feature group function
            memory (int): Memory (in GB) for the feature group function
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency

        Returns:
            FeatureGroup: The updated feature group"""
        return self._call_api('updateFeatureGroupFunctionDefinition', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'functionSourceCode': function_source_code, 'functionName': function_name, 'inputFeatureGroups': input_feature_groups, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements}, parse_type=FeatureGroup)

    def update_feature_group_zip(self, feature_group_id: str, function_name: str, module_name: str, input_feature_groups: list = None, cpu_size: str = None, memory: int = None, package_requirements: dict = None) -> Upload:
        """Updates the zip for a feature group created using createFeatureGroupFromZip

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            function_name (str): Name of the function found in the source code that will be executed (on the optional inputs) to materialize this feature group.
            module_name (str): Path to the file with the feature group function.
            input_feature_groups (list): List of feature groups that are supplied to the function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            cpu_size (str): Size of the cpu for the feature group function
            memory (int): Memory (in GB) for the feature group function
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency

        Returns:
            Upload: The Upload to upload the zip file to"""
        return self._call_api('updateFeatureGroupZip', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'functionName': function_name, 'moduleName': module_name, 'inputFeatureGroups': input_feature_groups, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements}, parse_type=Upload)

    def update_feature_group_git(self, feature_group_id: str, application_connector_id: str = None, branch_name: str = None, python_root: str = None, function_name: str = None, module_name: str = None, input_feature_groups: list = None, cpu_size: str = None, memory: int = None, package_requirements: dict = None) -> FeatureGroup:
        """Updates a feature group created using createFeatureGroupFromGit

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            application_connector_id (str): The unique ID associated with the git application connector.
            branch_name (str): Name of the branch in the git repository to be used for training.
            python_root (str): Path from the top level of the git repository to the directory containing the Python source code. If not provided, the default is the root of the git repository.
            function_name (str): Name of the function found in the source code that will be executed (on the optional inputs) to materialize this feature group.
            module_name (str): Path to the file with the feature group function.
            input_feature_groups (list): List of feature groups that are supplied to the function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            cpu_size (str): Size of the cpu for the feature group function
            memory (int): Memory (in GB) for the feature group function
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency

        Returns:
            FeatureGroup: The updated FeatureGroup"""
        return self._call_api('updateFeatureGroupGit', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'applicationConnectorId': application_connector_id, 'branchName': branch_name, 'pythonRoot': python_root, 'functionName': function_name, 'moduleName': module_name, 'inputFeatureGroups': input_feature_groups, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements}, parse_type=FeatureGroup)

    def update_feature(self, feature_group_id: str, name: str, select_expression: str = None, new_name: str = None) -> FeatureGroup:
        """Modifies an existing feature in a feature group. A user needs to specify the name and feature group ID and either a SQL statement or new name tp update the feature.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            name (str): The name of the feature to be updated.
            select_expression (str): Input SQL statement for modifying the feature.
            new_name (str):  The new name of the feature.

        Returns:
            FeatureGroup: The updated feature group object."""
        return self._call_api('updateFeature', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'name': name, 'selectExpression': select_expression, 'newName': new_name}, parse_type=FeatureGroup)

    def export_feature_group_version_to_file_connector(self, feature_group_version: str, location: str, export_file_format: str, overwrite: bool = False) -> FeatureGroupExport:
        """Export Feature group to File Connector.

        Args:
            feature_group_version (str): The Feature Group instance to export.
            location (str): Cloud file location to export to.
            export_file_format (str): File format to export to.
            overwrite (bool): If true and a file exists at this location, this process will overwrite the file.

        Returns:
            FeatureGroupExport: The FeatureGroupExport instance"""
        return self._call_api('exportFeatureGroupVersionToFileConnector', 'POST', query_params={}, body={'featureGroupVersion': feature_group_version, 'location': location, 'exportFileFormat': export_file_format, 'overwrite': overwrite}, parse_type=FeatureGroupExport)

    def export_feature_group_version_to_database_connector(self, feature_group_version: str, database_connector_id: str, object_name: str, write_mode: str, database_feature_mapping: dict, id_column: str = None, additional_id_columns: list = None) -> FeatureGroupExport:
        """Export Feature group to Database Connector.

        Args:
            feature_group_version (str): The Feature Group instance id to export.
            database_connector_id (str): Database connector to export to.
            object_name (str): The database object to write to
            write_mode (str): Either INSERT or UPSERT
            database_feature_mapping (dict): A key/value pair JSON Object of "database connector column" -> "feature name" pairs.
            id_column (str): Required if mode is UPSERT. Indicates which database column should be used as the lookup key for UPSERT
            additional_id_columns (list): For database connectors which support it, additional ID columns to use as a complex key for upserting

        Returns:
            FeatureGroupExport: The FeatureGroupExport instance"""
        return self._call_api('exportFeatureGroupVersionToDatabaseConnector', 'POST', query_params={}, body={'featureGroupVersion': feature_group_version, 'databaseConnectorId': database_connector_id, 'objectName': object_name, 'writeMode': write_mode, 'databaseFeatureMapping': database_feature_mapping, 'idColumn': id_column, 'additionalIdColumns': additional_id_columns}, parse_type=FeatureGroupExport)

    def export_feature_group_version_to_console(self, feature_group_version: str, export_file_format: str) -> FeatureGroupExport:
        """Export Feature group to console.

        Args:
            feature_group_version (str): The Feature Group instance to export.
            export_file_format (str): File format to export to.

        Returns:
            FeatureGroupExport: The FeatureGroupExport instance"""
        return self._call_api('exportFeatureGroupVersionToConsole', 'POST', query_params={}, body={'featureGroupVersion': feature_group_version, 'exportFileFormat': export_file_format}, parse_type=FeatureGroupExport)

    def set_feature_group_modifier_lock(self, feature_group_id: str, locked: bool = True):
        """To lock a feature group to prevent it from being modified.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            locked (bool): True or False to disable or enable feature group modification."""
        return self._call_api('setFeatureGroupModifierLock', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'locked': locked})

    def add_user_to_feature_group_modifiers(self, feature_group_id: str, email: str):
        """Adds user to a feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            email (str): The email address of the user to be removed."""
        return self._call_api('addUserToFeatureGroupModifiers', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'email': email})

    def add_organization_group_to_feature_group_modifiers(self, feature_group_id: str, organization_group_id: str):
        """Add Organization to a feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            organization_group_id (str): The unique ID associated with the organization group."""
        return self._call_api('addOrganizationGroupToFeatureGroupModifiers', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'organizationGroupId': organization_group_id})

    def remove_user_from_feature_group_modifiers(self, feature_group_id: str, email: str):
        """Removes user from a feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            email (str): The email address of the user to be removed."""
        return self._call_api('removeUserFromFeatureGroupModifiers', 'DELETE', query_params={'featureGroupId': feature_group_id, 'email': email})

    def remove_organization_group_from_feature_group_modifiers(self, feature_group_id: str, organization_group_id: str):
        """Removes Organization from a feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            organization_group_id (str): The unique ID associated with the organization group."""
        return self._call_api('removeOrganizationGroupFromFeatureGroupModifiers', 'DELETE', query_params={'featureGroupId': feature_group_id, 'organizationGroupId': organization_group_id})

    def delete_feature(self, feature_group_id: str, name: str) -> FeatureGroup:
        """Removes an existing feature from a feature group. A user needs to specify the name of the feature to be deleted and the feature group ID.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            name (str): The name of the feature to be deleted.

        Returns:
            FeatureGroup: The updated feature group object."""
        return self._call_api('deleteFeature', 'DELETE', query_params={'featureGroupId': feature_group_id, 'name': name}, parse_type=FeatureGroup)

    def delete_feature_group(self, feature_group_id: str):
        """Removes an existing feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group."""
        return self._call_api('deleteFeatureGroup', 'DELETE', query_params={'featureGroupId': feature_group_id})

    def create_feature_group_version(self, feature_group_id: str, variable_bindings: dict = None) -> FeatureGroupVersion:
        """Creates a snapshot for a specified feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            variable_bindings (dict): (JSON Object): JSON object (aka map) defining variable bindings that override parent feature group values.

        Returns:
            FeatureGroupVersion: A feature group version."""
        return self._call_api('createFeatureGroupVersion', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'variableBindings': variable_bindings}, parse_type=FeatureGroupVersion)

    def create_feature_group_template(self, feature_group_id: str, name: str, template_sql: str, template_variables: list, description: str = None, template_bindings: list = None, should_attach_feature_group_to_template: bool = False) -> FeatureGroupTemplate:
        """Create a feature group template.

        Args:
            feature_group_id (str): Identifier of the feature group this template was created from.
            name (str): The user-friendly of for this feature group template.
            template_sql (str): The template sql that will be resolved by applying values from the template variables to generate sql for a feature group.
            template_variables (list): The template variables for resolving the template.
            description (str): A description of this feature group template
            template_bindings (list): If the feature group will be attached to the newly created template, set these variable bindings on that feature group.
            should_attach_feature_group_to_template (bool): Set to True to convert the feature group to a template feature group and attach it to the newly created template.

        Returns:
            FeatureGroupTemplate: The created feature group template"""
        return self._call_api('createFeatureGroupTemplate', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'name': name, 'templateSql': template_sql, 'templateVariables': template_variables, 'description': description, 'templateBindings': template_bindings, 'shouldAttachFeatureGroupToTemplate': should_attach_feature_group_to_template}, parse_type=FeatureGroupTemplate)

    def delete_feature_group_template(self, feature_group_template_id: str):
        """Delete an existing feature group template.

        Args:
            feature_group_template_id (str): The unique ID associated with the feature group template."""
        return self._call_api('deleteFeatureGroupTemplate', 'DELETE', query_params={'featureGroupTemplateId': feature_group_template_id})

    def update_feature_group_template(self, feature_group_template_id: str, template_sql: str = None, template_variables: list = None, description: str = None, name: str = None) -> FeatureGroupTemplate:
        """Update a feature group template.

        Args:
            feature_group_template_id (str): Identifier of the feature group template to update.
            template_sql (str): If provided, the new value to use for the template sql.
            template_variables (list): If provided, the new value to use for the template variables.
            description (str): A description of this feature group template
            name (str): The user-friendly of for this feature group template.

        Returns:
            FeatureGroupTemplate: The updated feature group template."""
        return self._call_api('updateFeatureGroupTemplate', 'POST', query_params={}, body={'featureGroupTemplateId': feature_group_template_id, 'templateSql': template_sql, 'templateVariables': template_variables, 'description': description, 'name': name}, parse_type=FeatureGroupTemplate)

    def preview_feature_group_template_resolution(self, feature_group_template_id: str = None, template_bindings: list = None, template_sql: str = None, template_variables: list = None, should_validate: bool = True) -> ResolvedFeatureGroupTemplate:
        """Resolve template sql using template variables and template bindings.

        Args:
            feature_group_template_id (str): If specified, use this template, otherwise assume an empty template.
            template_bindings (list): Values that overide the template variable values specified by the template.
            template_sql (str): If specified, use this as the template sql instead of the feature group template's sql.
            template_variables (list): Template variables to use. If a template is provided, this overrides the template's template variables.
            should_validate (bool): 

        Returns:
            ResolvedFeatureGroupTemplate: None"""
        return self._call_api('previewFeatureGroupTemplateResolution', 'POST', query_params={}, body={'featureGroupTemplateId': feature_group_template_id, 'templateBindings': template_bindings, 'templateSql': template_sql, 'templateVariables': template_variables, 'shouldValidate': should_validate}, parse_type=ResolvedFeatureGroupTemplate)

    def cancel_upload(self, upload_id: str):
        """Cancels an upload

        Args:
            upload_id (str): The Upload ID"""
        return self._call_api('cancelUpload', 'DELETE', query_params={'uploadId': upload_id})

    def upload_part(self, upload_id: str, part_number: int, part_data: io.TextIOBase) -> UploadPart:
        """Uploads a part of a large dataset file from your bucket to our system. Our system currently supports a size of up to 5GB for a part of a full file and a size of up to 5TB for the full file. Note that each part must be >=5MB in size, unless it is the last part in the sequence of parts for the full file.

        Args:
            upload_id (str): A unique identifier for this upload
            part_number (int): The 1-indexed number denoting the position of the file part in the sequence of parts for the full file.
            part_data (io.TextIOBase): The multipart/form-data for the current part of the full file.

        Returns:
            UploadPart: The object 'UploadPart' which encapsulates the hash and the etag for the part that got uploaded."""
        return self._call_api('uploadPart', 'POST', query_params={'uploadId': upload_id, 'partNumber': part_number}, parse_type=UploadPart, files={'partData': part_data})

    def mark_upload_complete(self, upload_id: str) -> Upload:
        """Marks an upload process as complete.

        Args:
            upload_id (str): A unique identifier for this upload

        Returns:
            Upload: The upload object associated with the upload process for the full file. The details of the object are described below:"""
        return self._call_api('markUploadComplete', 'POST', query_params={}, body={'uploadId': upload_id}, parse_type=Upload)

    def create_dataset_from_file_connector(self, name: str, table_name: str, location: str, file_format: str = None, refresh_schedule: str = None, csv_delimiter: str = None, filename_column: str = None, start_prefix: str = None, until_prefix: str = None, location_date_format: str = None, date_format_lookback_days: int = None, incremental: bool = False) -> Dataset:
        """Creates a dataset from a file located in a cloud storage, such as Amazon AWS S3, using the specified dataset name and location.

        Args:
            name (str): The name for the dataset.
            table_name (str): Organization-unique table name or the name of the feature group table to create using the source table.
            location (str): The URI location format of the dataset source. The URI location format needs to be specified to match the location_date_format when location_date_format is specified. Ex. Location = s3://bucket1/dir1/dir2/event_date=YYYY-MM-DD/* when The URI location format needs to include both the start_prefix and until_prefix when both are specified. Ex. Location s3://bucket1/dir1/* includes both s3://bucket1/dir1/dir2/event_date=2021-08-02/* and s3://bucket1/dir1/dir2/event_date=2021-08-08/*
            file_format (str): The file format of the dataset.
            refresh_schedule (str): The Cron time string format that describes a schedule to retrieve the latest version of the imported dataset. The time is specified in UTC.
            csv_delimiter (str): If the file format is CSV, use a specific csv delimiter.
            filename_column (str): Adds a new column to the dataset with the external URI path.
            start_prefix (str): The start prefix (inclusive) for a range based search on a cloud storage location URI.
            until_prefix (str): The end prefix (exclusive) for a range based search on a cloud storage location URI.
            location_date_format (str): The date format in which the data is partitioned in the cloud storage location. E.g., if the data is partitioned as s3://bucket1/dir1/dir2/event_date=YYYY-MM-DD/dir4/filename.parquet, then the location_date_format is YYYY-MM-DD This format needs to be consistent across all files within the specified location.
            date_format_lookback_days (int): The number of days to look back from the current day for import locations that are date partitioned. E.g., import date, 2021-06-04, with date_format_lookback_days = 3 will retrieve data for all the dates in the range [2021-06-02, 2021-06-04].
            incremental (bool): Signifies if the dataset is an incremental dataset.

        Returns:
            Dataset: The dataset created."""
        return self._call_api('createDatasetFromFileConnector', 'POST', query_params={}, body={'name': name, 'tableName': table_name, 'location': location, 'fileFormat': file_format, 'refreshSchedule': refresh_schedule, 'csvDelimiter': csv_delimiter, 'filenameColumn': filename_column, 'startPrefix': start_prefix, 'untilPrefix': until_prefix, 'locationDateFormat': location_date_format, 'dateFormatLookbackDays': date_format_lookback_days, 'incremental': incremental}, parse_type=Dataset)

    def create_dataset_version_from_file_connector(self, dataset_id: str, location: str = None, file_format: str = None, csv_delimiter: str = None) -> DatasetVersion:
        """Creates a new version of the specified dataset.

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            location (str): A new external URI to import the dataset from. If not specified, the last location will be used.
            file_format (str): The fileFormat to be used. If not specified, the service will try to detect the file format.
            csv_delimiter (str): If the file format is CSV, use a specific csv delimiter.

        Returns:
            DatasetVersion: The new Dataset Version created."""
        return self._call_api('createDatasetVersionFromFileConnector', 'POST', query_params={'datasetId': dataset_id}, body={'location': location, 'fileFormat': file_format, 'csvDelimiter': csv_delimiter}, parse_type=DatasetVersion)

    def create_dataset_from_database_connector(self, name: str, table_name: str, database_connector_id: str, object_name: str = None, columns: str = None, query_arguments: str = None, refresh_schedule: str = None, sql_query: str = None, incremental: bool = False, timestamp_column: str = None) -> Dataset:
        """Creates a dataset from a Database Connector

        Args:
            name (str): The name for the dataset to be attached.
            table_name (str): Organization-unique table name
            database_connector_id (str): The Database Connector to import the dataset from
            object_name (str): If applicable, the name/id of the object in the service to query.
            columns (str): The columns to query from the external service object.
            query_arguments (str): Additional query arguments to filter the data
            refresh_schedule (str): The Cron time string format that describes a schedule to retrieve the latest version of the imported dataset. The time is specified in UTC.
            sql_query (str): The full SQL query to use when fetching data. If present, this parameter will override objectName, columns, timestampColumn, and queryArguments
            incremental (bool): Signifies if the dataset is an incremental dataset.
            timestamp_column (str): If dataset is incremental, this is the column name of the required column in the dataset. This column must contain timestamps in descending order which are used to determine the increments of the incremental dataset.

        Returns:
            Dataset: The created dataset."""
        return self._call_api('createDatasetFromDatabaseConnector', 'POST', query_params={}, body={'name': name, 'tableName': table_name, 'databaseConnectorId': database_connector_id, 'objectName': object_name, 'columns': columns, 'queryArguments': query_arguments, 'refreshSchedule': refresh_schedule, 'sqlQuery': sql_query, 'incremental': incremental, 'timestampColumn': timestamp_column}, parse_type=Dataset)

    def create_dataset_from_application_connector(self, name: str, table_name: str, application_connector_id: str, object_id: str = None, start_timestamp: int = None, end_timestamp: int = None, refresh_schedule: str = None) -> Dataset:
        """Creates a dataset from an Application Connector

        Args:
            name (str): The name for the dataset
            table_name (str): Organization-unique table name
            application_connector_id (str): The unique application connector to download data from
            object_id (str): If applicable, the id of the object in the service to query.
            start_timestamp (int): The Unix timestamp of the start of the period that will be queried.
            end_timestamp (int): The Unix timestamp of the end of the period that will be queried.
            refresh_schedule (str): The Cron time string format that describes a schedule to retrieve the latest version of the imported dataset. The time is specified in UTC.

        Returns:
            Dataset: The created dataset."""
        return self._call_api('createDatasetFromApplicationConnector', 'POST', query_params={}, body={'name': name, 'tableName': table_name, 'applicationConnectorId': application_connector_id, 'objectId': object_id, 'startTimestamp': start_timestamp, 'endTimestamp': end_timestamp, 'refreshSchedule': refresh_schedule}, parse_type=Dataset)

    def create_dataset_version_from_database_connector(self, dataset_id: str, object_name: str = None, columns: str = None, query_arguments: str = None, sql_query: str = None) -> DatasetVersion:
        """Creates a new version of the specified dataset

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            object_name (str): If applicable, the name/id of the object in the service to query. If not specified, the last name will be used.
            columns (str): The columns to query from the external service object. If not specified, the last columns will be used.
            query_arguments (str): Additional query arguments to filter the data. If not specified, the last arguments will be used.
            sql_query (str): The full SQL query to use when fetching data. If present, this parameter will override objectName, columns, and queryArguments

        Returns:
            DatasetVersion: The new Dataset Version created."""
        return self._call_api('createDatasetVersionFromDatabaseConnector', 'POST', query_params={'datasetId': dataset_id}, body={'objectName': object_name, 'columns': columns, 'queryArguments': query_arguments, 'sqlQuery': sql_query}, parse_type=DatasetVersion)

    def create_dataset_version_from_application_connector(self, dataset_id: str, object_id: str = None, start_timestamp: int = None, end_timestamp: int = None) -> DatasetVersion:
        """Creates a new version of the specified dataset

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            object_id (str): If applicable, the id of the object in the service to query. If not specified, the last name will be used.
            start_timestamp (int): The Unix timestamp of the start of the period that will be queried.
            end_timestamp (int): The Unix timestamp of the end of the period that will be queried.

        Returns:
            DatasetVersion: The new Dataset Version created."""
        return self._call_api('createDatasetVersionFromApplicationConnector', 'POST', query_params={'datasetId': dataset_id}, body={'objectId': object_id, 'startTimestamp': start_timestamp, 'endTimestamp': end_timestamp}, parse_type=DatasetVersion)

    def create_dataset_from_upload(self, name: str, table_name: str, file_format: str = None, csv_delimiter: str = None) -> Upload:
        """Creates a dataset and return an upload Id that can be used to upload a file.

        Args:
            name (str): The name for the dataset.
            table_name (str): Organization-unique table name for this dataset.
            file_format (str): The file format of the dataset.
            csv_delimiter (str): If the file format is CSV, use a specific csv delimiter.

        Returns:
            Upload: A refernce to be used when uploading file parts."""
        return self._call_api('createDatasetFromUpload', 'POST', query_params={}, body={'name': name, 'tableName': table_name, 'fileFormat': file_format, 'csvDelimiter': csv_delimiter}, parse_type=Upload)

    def create_dataset_version_from_upload(self, dataset_id: str, file_format: str = None) -> Upload:
        """Creates a new version of the specified dataset using a local file upload.

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            file_format (str): The file_format to be used. If not specified, the service will try to detect the file format.

        Returns:
            Upload: A token to be used when uploading file parts."""
        return self._call_api('createDatasetVersionFromUpload', 'POST', query_params={'datasetId': dataset_id}, body={'fileFormat': file_format}, parse_type=Upload)

    def create_streaming_dataset(self, name: str, table_name: str, project_id: str = None, dataset_type: str = None) -> Dataset:
        """Creates a streaming dataset. Use a streaming dataset if your dataset is receiving information from multiple sources over an extended period of time.

        Args:
            name (str): The name for the dataset.
            table_name (str): The feature group table name to create for this dataset
            project_id (str): The project to create the streaming dataset for.
            dataset_type (str): The dataset has to be a type that is associated with the use case of your project. Please see (Use Case Documentation)[https://api.abacus.ai/app/help/useCases] for the datasetTypes that are supported per use case.

        Returns:
            Dataset: The streaming dataset created."""
        return self._call_api('createStreamingDataset', 'POST', query_params={}, body={'name': name, 'tableName': table_name, 'projectId': project_id, 'datasetType': dataset_type}, parse_type=Dataset)

    def snapshot_streaming_data(self, dataset_id: str) -> DatasetVersion:
        """Snapshots the current data in the streaming dataset for training.

        Args:
            dataset_id (str): The unique ID associated with the dataset.

        Returns:
            DatasetVersion: The new Dataset Version created."""
        return self._call_api('snapshotStreamingData', 'POST', query_params={'datasetId': dataset_id}, body={}, parse_type=DatasetVersion)

    def set_dataset_column_data_type(self, dataset_id: str, column: str, data_type: str) -> Dataset:
        """Set a column's type in a specified dataset.

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            column (str): The name of the column.
            data_type (str): The type of the data in the column.  INTEGER,  FLOAT,  STRING,  DATE,  DATETIME,  BOOLEAN,  LIST,  STRUCT Refer to the (guide on data types)[https://api.abacus.ai/app/help/class/DataType] for more information. Note: Some ColumnMappings will restrict the options or explicity set the DataType.

        Returns:
            Dataset: The dataset and schema after the data_type has been set"""
        return self._call_api('setDatasetColumnDataType', 'POST', query_params={'datasetId': dataset_id}, body={'column': column, 'dataType': data_type}, parse_type=Dataset)

    def create_dataset_from_streaming_connector(self, name: str, table_name: str, streaming_connector_id: str, streaming_args: dict = None, refresh_schedule: str = None) -> Dataset:
        """Creates a dataset from a Streaming Connector

        Args:
            name (str): The name for the dataset to be attached.
            table_name (str): Organization-unique table name
            streaming_connector_id (str): The Streaming Connector to import the dataset from
            streaming_args (dict): Dict of arguments to read data from the streaming connector
            refresh_schedule (str): The Cron time string format that describes a schedule to retrieve the latest version of the imported dataset. The time is specified in UTC.

        Returns:
            Dataset: The created dataset."""
        return self._call_api('createDatasetFromStreamingConnector', 'POST', query_params={}, body={'name': name, 'tableName': table_name, 'streamingConnectorId': streaming_connector_id, 'streamingArgs': streaming_args, 'refreshSchedule': refresh_schedule}, parse_type=Dataset)

    def set_streaming_retention_policy(self, dataset_id: str, retention_hours: int = None, retention_row_count: int = None):
        """Sets the streaming retention policy

        Args:
            dataset_id (str): The Streaming dataset
            retention_hours (int): The number of hours to retain streamed data in memory
            retention_row_count (int): The number of rows to retain streamed data in memory"""
        return self._call_api('setStreamingRetentionPolicy', 'POST', query_params={'datasetId': dataset_id}, body={'retentionHours': retention_hours, 'retentionRowCount': retention_row_count})

    def rename_database_connector(self, database_connector_id: str, name: str):
        """Renames a Database Connector

        Args:
            database_connector_id (str): The unique identifier for the database connector.
            name (str): The new name for the Database Connector"""
        return self._call_api('renameDatabaseConnector', 'PATCH', query_params={}, body={'databaseConnectorId': database_connector_id, 'name': name})

    def rename_application_connector(self, application_connector_id: str, name: str):
        """Renames an Application Connector

        Args:
            application_connector_id (str): The unique identifier for the application connector.
            name (str): A new name for the application connector"""
        return self._call_api('renameApplicationConnector', 'PATCH', query_params={}, body={'applicationConnectorId': application_connector_id, 'name': name})

    def verify_database_connector(self, database_connector_id: str):
        """Checks to see if Abacus.AI can access the database.

        Args:
            database_connector_id (str): The unique identifier for the database connector."""
        return self._call_api('verifyDatabaseConnector', 'POST', query_params={}, body={'databaseConnectorId': database_connector_id})

    def verify_file_connector(self, bucket: str) -> FileConnectorVerification:
        """Checks to see if Abacus.AI can access the bucket.

        Args:
            bucket (str): The bucket to test.

        Returns:
            FileConnectorVerification: The Result of the verification."""
        return self._call_api('verifyFileConnector', 'POST', query_params={}, body={'bucket': bucket}, parse_type=FileConnectorVerification)

    def delete_database_connector(self, database_connector_id: str):
        """Delete a database connector.

        Args:
            database_connector_id (str): The unique identifier for the database connector."""
        return self._call_api('deleteDatabaseConnector', 'DELETE', query_params={'databaseConnectorId': database_connector_id})

    def delete_application_connector(self, application_connector_id: str):
        """Delete a application connector.

        Args:
            application_connector_id (str): The unique identifier for the application connector."""
        return self._call_api('deleteApplicationConnector', 'DELETE', query_params={'applicationConnectorId': application_connector_id})

    def delete_file_connector(self, bucket: str):
        """Removes a connected service from the specified organization.

        Args:
            bucket (str): The fully qualified URI of the bucket to remove."""
        return self._call_api('deleteFileConnector', 'DELETE', query_params={'bucket': bucket})

    def verify_application_connector(self, application_connector_id: str):
        """Checks to see if Abacus.AI can access the Application.

        Args:
            application_connector_id (str): The unique identifier for the application connector."""
        return self._call_api('verifyApplicationConnector', 'POST', query_params={}, body={'applicationConnectorId': application_connector_id})

    def set_azure_blob_connection_string(self, bucket: str, connection_string: str) -> FileConnectorVerification:
        """Authenticates specified Azure Blob Storage bucket using an authenticated Connection String.

        Args:
            bucket (str): The fully qualified Azure Blob Storage Bucket URI
            connection_string (str): The Connection String {product_name} should use to authenticate when accessing this bucket

        Returns:
            FileConnectorVerification: An object with the roleArn and verification status for the specified bucket."""
        return self._call_api('setAzureBlobConnectionString', 'POST', query_params={}, body={'bucket': bucket, 'connectionString': connection_string}, parse_type=FileConnectorVerification)

    def verify_streaming_connector(self, streaming_connector_id: str):
        """Checks to see if Abacus.AI can access the streaming connector.

        Args:
            streaming_connector_id (str): The unique identifier for the streaming connector."""
        return self._call_api('verifyStreamingConnector', 'POST', query_params={}, body={'streamingConnectorId': streaming_connector_id})

    def rename_streaming_connector(self, streaming_connector_id: str, name: str):
        """Renames a Streaming Connector

        Args:
            streaming_connector_id (str): The unique identifier for the streaming connector.
            name (str): A new name for the streaming connector"""
        return self._call_api('renameStreamingConnector', 'PATCH', query_params={}, body={'streamingConnectorId': streaming_connector_id, 'name': name})

    def delete_streaming_connector(self, streaming_connector_id: str):
        """Delete a streaming connector.

        Args:
            streaming_connector_id (str): The unique identifier for the streaming connector."""
        return self._call_api('deleteStreamingConnector', 'DELETE', query_params={'streamingConnectorId': streaming_connector_id})

    def create_streaming_token(self) -> StreamingAuthToken:
        """Creates a streaming token for the specified project. Streaming tokens are used to authenticate requests to append data to streaming datasets.

        Returns:
            StreamingAuthToken: The streaming token."""
        return self._call_api('createStreamingToken', 'POST', query_params={}, body={}, parse_type=StreamingAuthToken)

    def delete_streaming_token(self, streaming_token: str):
        """Deletes the specified streaming token.

        Args:
            streaming_token (str): The streaming token to delete."""
        return self._call_api('deleteStreamingToken', 'DELETE', query_params={'streamingToken': streaming_token})

    def attach_dataset_to_project(self, dataset_id: str, project_id: str, dataset_type: str) -> List[Schema]:
        """[DEPRECATED] Attaches the dataset to the project.

        Use this method to attach a dataset that is already in the organization to another project. The dataset type is required to let the AI engine know what type of schema should be used.


        Args:
            dataset_id (str): The dataset to attach.
            project_id (str): The project to attach the dataset to.
            dataset_type (str): The dataset has to be a type that is associated with the use case of your project. Please see (Use Case Documentation)[https://api.abacus.ai/app/help/useCases] for the datasetTypes that are supported per use case.

        Returns:
            Schema: An array of columns descriptions."""
        logging.warning(
            'This function (attachDatasetToProject) is deprecated and will be removed in a future version.')
        return self._call_api('attachDatasetToProject', 'POST', query_params={'datasetId': dataset_id}, body={'projectId': project_id, 'datasetType': dataset_type}, parse_type=Schema)

    def remove_dataset_from_project(self, dataset_id: str, project_id: str):
        """[DEPRECATED] Removes a dataset from a project.

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            project_id (str): The unique ID associated with the project."""
        logging.warning(
            'This function (removeDatasetFromProject) is deprecated and will be removed in a future version.')
        return self._call_api('removeDatasetFromProject', 'POST', query_params={'datasetId': dataset_id}, body={'projectId': project_id})

    def rename_dataset(self, dataset_id: str, name: str):
        """Rename a dataset.

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            name (str): The new name for the dataset."""
        return self._call_api('renameDataset', 'PATCH', query_params={'datasetId': dataset_id}, body={'name': name})

    def delete_dataset(self, dataset_id: str):
        """Deletes the specified dataset from the organization.

        The dataset cannot be deleted if it is currently attached to a project.


        Args:
            dataset_id (str): The dataset to delete."""
        return self._call_api('deleteDataset', 'DELETE', query_params={'datasetId': dataset_id})

    def train_model(self, project_id: str, name: str = None, training_config: dict = None, feature_group_ids: list = None, refresh_schedule: str = None, custom_algorithms: list = None, custom_algorithms_only: bool = False, custom_algorithm_configs: dict = None, cpu_size: str = None, memory: int = None) -> Model:
        """Trains a model for the specified project.

        Use this method to train a model in this project. This method supports user-specified training configurations defined in the getTrainingConfigOptions method.


        Args:
            project_id (str): The unique ID associated with the project.
            name (str): The name you want your model to have. Defaults to "<Project Name> Model".
            training_config (dict): The training config key/value pairs used to train this model.
            feature_group_ids (list): List of feature group ids provided by the user to train the model on.
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically retrain the created model.
            custom_algorithms (list): List of user-defined algorithms to train.
            custom_algorithms_only (bool): Whether only run custom algorithms.
            custom_algorithm_configs (dict): Configs for each user-defined algorithm, key is algorithm name, value is the config serialized to json
            cpu_size (str): Size of the cpu for the user-defined algorithms during train.
            memory (int): Memory (in GB) for the user-defined algorithms during train.

        Returns:
            Model: The new model which is being trained."""
        return self._call_api('trainModel', 'POST', query_params={}, body={'projectId': project_id, 'name': name, 'trainingConfig': training_config, 'featureGroupIds': feature_group_ids, 'refreshSchedule': refresh_schedule, 'customAlgorithms': custom_algorithms, 'customAlgorithmsOnly': custom_algorithms_only, 'customAlgorithmConfigs': custom_algorithm_configs, 'cpuSize': cpu_size, 'memory': memory}, parse_type=Model)

    def create_model_from_python(self, project_id: str, function_source_code: str, train_function_name: str, training_input_tables: list, predict_function_name: str = None, predict_many_function_name: str = None, initialize_function_name: str = None, name: str = None, cpu_size: str = None, memory: int = None, training_config: dict = None, exclusive_run: bool = False, package_requirements: dict = None) -> Model:
        """Initializes a new Model from user provided Python code. If a list of input feature groups are supplied,

        we will provide as arguments to the train and predict functions with the materialized feature groups for those
        input feature groups.

        This method expects `functionSourceCode` to be a valid language source file which contains the functions named
        `trainFunctionName` and `predictFunctionName`. `trainFunctionName` returns the ModelVersion that is the result of
        training the model using `trainFunctionName` and `predictFunctionName` has no well defined return type,
        as it returns the prediction made by the `predictFunctionName`, which can be anything


        Args:
            project_id (str): The unique ID associated with the project.
            function_source_code (str): Contents of a valid python source code file. The source code should contain the functions named trainFunctionName and predictFunctionName. A list of allowed import and system libraries for each language is specified in the user functions documentation section.
            train_function_name (str): Name of the function found in the source code that will be executed to train the model. It is not executed when this function is run.
            training_input_tables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            predict_function_name (str): Name of the function found in the source code that will be executed run predictions through model. It is not executed when this function is run.
            predict_many_function_name (str): Name of the function found in the source code that will be executed for batch prediction of the model. It is not executed when this function is run.
            initialize_function_name (str): Name of the function found in the source code to initialize the trained model before using it to make predictions using the model
            name (str): The name you want your model to have. Defaults to "<Project Name> Model"
            cpu_size (str): Size of the cpu for the model training function
            memory (int): Memory (in GB) for the model training function
            training_config (dict): Training configuration
            exclusive_run (bool): Decides if this model will be run exclusively OR along with other Abacus.ai algorithms
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency

        Returns:
            Model: The new model, which has not been trained."""
        return self._call_api('createModelFromPython', 'POST', query_params={}, body={'projectId': project_id, 'functionSourceCode': function_source_code, 'trainFunctionName': train_function_name, 'trainingInputTables': training_input_tables, 'predictFunctionName': predict_function_name, 'predictManyFunctionName': predict_many_function_name, 'initializeFunctionName': initialize_function_name, 'name': name, 'cpuSize': cpu_size, 'memory': memory, 'trainingConfig': training_config, 'exclusiveRun': exclusive_run, 'packageRequirements': package_requirements}, parse_type=Model)

    def create_model_from_zip(self, project_id: str, train_function_name: str, train_module_name: str, predict_module_name: str, training_input_tables: list, predict_function_name: str = None, predict_many_function_name: str = None, name: str = None, cpu_size: str = None, memory: int = None, package_requirements: dict = None) -> Upload:
        """Initializes a new Model from a user provided zip file containing Python code. If a list of input feature groups are supplied,

        we will provide as arguments to the train and predict functions with the materialized feature groups for those
        input feature groups.

        This method expects `trainModuleName` and `predictModuleName` to be valid language source files which contains the functions named
        `trainFunctionName` and `predictFunctionName`, respectively. `trainFunctionName` returns the ModelVersion that is the result of
        training the model using `trainFunctionName` and `predictFunctionName` has no well defined return type,
        as it returns the prediction made by the `predictFunctionName`, which can be anything


        Args:
            project_id (str): The unique ID associated with the project.
            train_function_name (str): Name of the function found in train module that will be executed to train the model. It is not executed when this function is run.
            train_module_name (str): Full path of the module that contains the train function from the root of the zip.
            predict_module_name (str): Full path of the module that contains the predict function from the root of the zip.
            training_input_tables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            predict_function_name (str): Name of the function found in the predict module that will be executed run predictions through model. It is not executed when this function is run.
            predict_many_function_name (str): Name of the function found in the predict module that will be executed run batch predictions through model. It is not executed when this function is run.
            name (str): The name you want your model to have. Defaults to "<Project Name> Model".
            cpu_size (str): Size of the cpu for the model training function
            memory (int): Memory (in GB) for the model training function
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency

        Returns:
            Upload: None"""
        return self._call_api('createModelFromZip', 'POST', query_params={}, body={'projectId': project_id, 'trainFunctionName': train_function_name, 'trainModuleName': train_module_name, 'predictModuleName': predict_module_name, 'trainingInputTables': training_input_tables, 'predictFunctionName': predict_function_name, 'predictManyFunctionName': predict_many_function_name, 'name': name, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements}, parse_type=Upload)

    def create_model_from_git(self, project_id: str, application_connector_id: str, branch_name: str, train_function_name: str, train_module_name: str, predict_module_name: str, training_input_tables: list, predict_function_name: str = None, predict_many_function_name: str = None, python_root: str = None, name: str = None, cpu_size: str = None, memory: int = None, package_requirements: dict = None) -> Model:
        """Initializes a new Model from a user provided git repository containing Python code. If a list of input feature groups are supplied,

        we will provide as arguments to the train and predict functions with the materialized feature groups for those
        input feature groups.

        This method expects `trainModuleName` and `predictModuleName` to be valid language source files which contains the functions named
        `trainFunctionName` and `predictFunctionName`, respectively. `trainFunctionName` returns the ModelVersion that is the result of
        training the model using `trainFunctionName` and `predictFunctionName` has no well defined return type,
        as it returns the prediction made by the `predictFunctionName`, which can be anything


        Args:
            project_id (str): The unique ID associated with the project.
            application_connector_id (str): The unique ID associated with the git application connector.
            branch_name (str): Name of the branch in the git repository to be used for training.
            train_function_name (str): Name of the function found in train module that will be executed to train the model. It is not executed when this function is run.
            train_module_name (str): Full path of the module that contains the train function from the root of the zip.
            predict_module_name (str): Full path of the module that contains the predict function from the root of the zip.
            training_input_tables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            predict_function_name (str): Name of the function found in the predict module that will be executed run predictions through model. It is not executed when this function is run.
            predict_many_function_name (str): 
            python_root (str): Path from the top level of the git repository to the directory containing the Python source code. If not provided, the default is the root of the git repository.
            name (str): The name you want your model to have. Defaults to "<Project Name> Model".
            cpu_size (str): Size of the cpu for the model training function
            memory (int): Memory (in GB) for the model training function
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency

        Returns:
            Model: None"""
        return self._call_api('createModelFromGit', 'POST', query_params={}, body={'projectId': project_id, 'applicationConnectorId': application_connector_id, 'branchName': branch_name, 'trainFunctionName': train_function_name, 'trainModuleName': train_module_name, 'predictModuleName': predict_module_name, 'trainingInputTables': training_input_tables, 'predictFunctionName': predict_function_name, 'predictManyFunctionName': predict_many_function_name, 'pythonRoot': python_root, 'name': name, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements}, parse_type=Model)

    def rename_model(self, model_id: str, name: str):
        """Renames a model

        Args:
            model_id (str): The ID of the model to rename
            name (str): The name to apply to the model"""
        return self._call_api('renameModel', 'PATCH', query_params={}, body={'modelId': model_id, 'name': name})

    def update_python_model(self, model_id: str, function_source_code: str = None, train_function_name: str = None, predict_function_name: str = None, predict_many_function_name: str = None, initialize_function_name: str = None, training_input_tables: list = None, cpu_size: str = None, memory: int = None, package_requirements: dict = None) -> Model:
        """Updates an existing python Model using user provided Python code. If a list of input feature groups are supplied,

        we will provide as arguments to the train and predict functions with the materialized feature groups for those
        input feature groups.

        This method expects `functionSourceCode` to be a valid language source file which contains the functions named
        `trainFunctionName` and `predictFunctionName`. `trainFunctionName` returns the ModelVersion that is the result of
        training the model using `trainFunctionName` and `predictFunctionName` has no well defined return type,
        as it returns the prediction made by the `predictFunctionName`, which can be anything


        Args:
            model_id (str): The unique ID associated with the Python model to be changed.
            function_source_code (str): Contents of a valid python source code file. The source code should contain the functions named trainFunctionName and predictFunctionName. A list of allowed import and system libraries for each language is specified in the user functions documentation section.
            train_function_name (str): Name of the function found in the source code that will be executed to train the model. It is not executed when this function is run.
            predict_function_name (str): Name of the function found in the source code that will be executed run predictions through model. It is not executed when this function is run.
            predict_many_function_name (str): Name of the function found in the source code that will be executed to run batch predictions through model. It is not executed when this function is run.
            initialize_function_name (str): Name of the function found in the source code to initialize the trained model before using it to make predictions using the model
            training_input_tables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            cpu_size (str): Size of the cpu for the model training function
            memory (int): Memory (in GB) for the model training function
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency

        Returns:
            Model: The updated model"""
        return self._call_api('updatePythonModel', 'POST', query_params={}, body={'modelId': model_id, 'functionSourceCode': function_source_code, 'trainFunctionName': train_function_name, 'predictFunctionName': predict_function_name, 'predictManyFunctionName': predict_many_function_name, 'initializeFunctionName': initialize_function_name, 'trainingInputTables': training_input_tables, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements}, parse_type=Model)

    def update_python_model_zip(self, model_id: str, train_function_name: str = None, predict_function_name: str = None, predict_many_function_name: str = None, train_module_name: str = None, predict_module_name: str = None, training_input_tables: list = None, cpu_size: str = None, memory: int = None, package_requirements: dict = None) -> Upload:
        """Updates an existing python Model using a provided zip file. If a list of input feature groups are supplied,

        we will provide as arguments to the train and predict functions with the materialized feature groups for those
        input feature groups.

        This method expects `trainModuleName` and `predictModuleName` to be valid language source files which contains the functions named
        `trainFunctionName` and `predictFunctionName`, respectively. `trainFunctionName` returns the ModelVersion that is the result of
        training the model using `trainFunctionName` and `predictFunctionName` has no well defined return type,
        as it returns the prediction made by the `predictFunctionName`, which can be anything


        Args:
            model_id (str): The unique ID associated with the Python model to be changed.
            train_function_name (str): Name of the function found in train module that will be executed to train the model. It is not executed when this function is run.
            predict_function_name (str): Name of the function found in the predict module that will be executed run predictions through model. It is not executed when this function is run.
            predict_many_function_name (str): Name of the function found in the predict module that will be executed run batch predictions through model. It is not executed when this function is run.
            train_module_name (str): Full path of the module that contains the train function from the root of the zip.
            predict_module_name (str): Full path of the module that contains the predict function from the root of the zip.
            training_input_tables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            cpu_size (str): Size of the cpu for the model training function
            memory (int): Memory (in GB) for the model training function
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency

        Returns:
            Upload: The updated model"""
        return self._call_api('updatePythonModelZip', 'POST', query_params={}, body={'modelId': model_id, 'trainFunctionName': train_function_name, 'predictFunctionName': predict_function_name, 'predictManyFunctionName': predict_many_function_name, 'trainModuleName': train_module_name, 'predictModuleName': predict_module_name, 'trainingInputTables': training_input_tables, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements}, parse_type=Upload)

    def update_python_model_git(self, model_id: str, application_connector_id: str = None, branch_name: str = None, python_root: str = None, train_function_name: str = None, predict_function_name: str = None, predict_many_function_name: str = None, train_module_name: str = None, predict_module_name: str = None, training_input_tables: list = None, cpu_size: str = None, memory: int = None) -> Model:
        """Updates an existing python Model using an existing git application connector. If a list of input feature groups are supplied,

        we will provide as arguments to the train and predict functions with the materialized feature groups for those
        input feature groups.

        This method expects `trainModuleName` and `predictModuleName` to be valid language source files which contains the functions named
        `trainFunctionName` and `predictFunctionName`, respectively. `trainFunctionName` returns the ModelVersion that is the result of
        training the model using `trainFunctionName` and `predictFunctionName` has no well defined return type,
        as it returns the prediction made by the `predictFunctionName`, which can be anything


        Args:
            model_id (str): The unique ID associated with the Python model to be changed.
            application_connector_id (str): The unique ID associated with the git application connector.
            branch_name (str): Name of the branch in the git repository to be used for training.
            python_root (str): Path from the top level of the git repository to the directory containing the Python source code. If not provided, the default is the root of the git repository.
            train_function_name (str): Name of the function found in train module that will be executed to train the model. It is not executed when this function is run.
            predict_function_name (str): Name of the function found in the predict module that will be executed run predictions through model. It is not executed when this function is run.
            predict_many_function_name (str): Name of the function found in the predict module that will be executed run batch predictions through model. It is not executed when this function is run.
            train_module_name (str): Full path of the module that contains the train function from the root of the zip.
            predict_module_name (str): Full path of the module that contains the predict function from the root of the zip.
            training_input_tables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            cpu_size (str): Size of the cpu for the model training function
            memory (int): Memory (in GB) for the model training function

        Returns:
            Model: The updated model"""
        return self._call_api('updatePythonModelGit', 'POST', query_params={}, body={'modelId': model_id, 'applicationConnectorId': application_connector_id, 'branchName': branch_name, 'pythonRoot': python_root, 'trainFunctionName': train_function_name, 'predictFunctionName': predict_function_name, 'predictManyFunctionName': predict_many_function_name, 'trainModuleName': train_module_name, 'predictModuleName': predict_module_name, 'trainingInputTables': training_input_tables, 'cpuSize': cpu_size, 'memory': memory}, parse_type=Model)

    def set_model_training_config(self, model_id: str, training_config: dict) -> Model:
        """Edits the default model training config

        Args:
            model_id (str): The unique ID of the model to update
            training_config (dict): The training config key/value pairs used to train this model.

        Returns:
            Model: The model object correspoding after the training config is applied"""
        return self._call_api('setModelTrainingConfig', 'PATCH', query_params={}, body={'modelId': model_id, 'trainingConfig': training_config}, parse_type=Model)

    def set_model_prediction_params(self, model_id: str, prediction_config: dict) -> Model:
        """Sets the model prediction config for the model

        Args:
            model_id (str): The unique ID of the model to update
            prediction_config (dict): The prediction config for the model

        Returns:
            Model: The model object correspoding after the prediction config is applied"""
        return self._call_api('setModelPredictionParams', 'PATCH', query_params={}, body={'modelId': model_id, 'predictionConfig': prediction_config}, parse_type=Model)

    def retrain_model(self, model_id: str, deployment_ids: list = [], feature_group_ids: list = None, custom_algorithm_configs: dict = None, cpu_size: str = None, memory: int = None) -> Model:
        """Retrains the specified model. Gives you an option to choose the deployments you want the retraining to be deployed to.

        Args:
            model_id (str): The model to retrain.
            deployment_ids (list): List of deployments to automatically deploy to.
            feature_group_ids (list): List of feature group ids provided by the user to train the model on.
            custom_algorithm_configs (dict): The user-defined training configs for each custom algorithm.
            cpu_size (str): Size of the cpu for the user-defined algorithms during train.
            memory (int): Memory (in GB) for the user-defined algorithms during train.

        Returns:
            Model: The model that is being retrained."""
        return self._call_api('retrainModel', 'POST', query_params={}, body={'modelId': model_id, 'deploymentIds': deployment_ids, 'featureGroupIds': feature_group_ids, 'customAlgorithmConfigs': custom_algorithm_configs, 'cpuSize': cpu_size, 'memory': memory}, parse_type=Model)

    def delete_model(self, model_id: str):
        """Deletes the specified model and all its versions. Models which are currently used in deployments cannot be deleted.

        Args:
            model_id (str): The ID of the model to delete."""
        return self._call_api('deleteModel', 'DELETE', query_params={'modelId': model_id})

    def delete_model_version(self, model_version: str):
        """Deletes the specified model version. Model Versions which are currently used in deployments cannot be deleted.

        Args:
            model_version (str): The ID of the model version to delete."""
        return self._call_api('deleteModelVersion', 'DELETE', query_params={'modelVersion': model_version})

    def export_model_artifact_as_feature_group(self, model_version: str, table_name: str, artifact_type: str) -> FeatureGroup:
        """Exports metric artifact data for a model as a feature group.

        Args:
            model_version (str): The version of the model.
            table_name (str): The name of the feature group table to create.
            artifact_type (str): An EvalArtifact enum of which artifact to export.

        Returns:
            FeatureGroup: The created feature group."""
        return self._call_api('exportModelArtifactAsFeatureGroup', 'POST', query_params={}, body={'modelVersion': model_version, 'tableName': table_name, 'artifactType': artifact_type}, parse_type=FeatureGroup)

    def get_custom_train_function_info(self, project_id: str, feature_group_names_for_training: list = None, training_data_parameter_name_override: dict = None, training_config: dict = None, custom_algorithm_config: dict = None) -> CustomTrainFunctionInfo:
        """Returns the information about how to call the custom train function.

        Args:
            project_id (str): The unique version ID of the project
            feature_group_names_for_training (list): A list of feature group table names that will be used for training
            training_data_parameter_name_override (dict): Override from feature group type to parameter name in train function.
            training_config (dict): Training config names to values for the options supported by Abacus.ai platform.
            custom_algorithm_config (dict): User-defined config that can be serialized by JSON.

        Returns:
            CustomTrainFunctionInfo: Information about how to call the customer provided train function."""
        return self._call_api('getCustomTrainFunctionInfo', 'POST', query_params={}, body={'projectId': project_id, 'featureGroupNamesForTraining': feature_group_names_for_training, 'trainingDataParameterNameOverride': training_data_parameter_name_override, 'trainingConfig': training_config, 'customAlgorithmConfig': custom_algorithm_config}, parse_type=CustomTrainFunctionInfo)

    def create_model_monitor(self, project_id: str, training_feature_group_id: str, prediction_feature_group_id: str, name: str = None, refresh_schedule: str = None, target_value: str = None, feature_mappings: dict = None, model_id: str = None, training_feature_mappings: dict = None) -> ModelMonitor:
        """Runs a model monitor for the specified project.

        Args:
            project_id (str): The unique ID associated with the project.
            training_feature_group_id (str): The unique ID of the training data feature group
            prediction_feature_group_id (str): The unique ID of the prediction data feature group
            name (str): The name you want your model monitor to have. Defaults to "<Project Name> Model Monitor".
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically retrain the created model monitor
            target_value (str): A target positive value for the label to compute bias for
            feature_mappings (dict): A json map to override features for prediction_feature_group, where keys are column names and the values are feature data use types.
            model_id (str): The Unique ID of the Model
            training_feature_mappings (dict): " A json map to override features for training_fature_group, where keys are column names and the values are feature data use types.

        Returns:
            ModelMonitor: The new model monitor that was created."""
        return self._call_api('createModelMonitor', 'POST', query_params={}, body={'projectId': project_id, 'trainingFeatureGroupId': training_feature_group_id, 'predictionFeatureGroupId': prediction_feature_group_id, 'name': name, 'refreshSchedule': refresh_schedule, 'targetValue': target_value, 'featureMappings': feature_mappings, 'modelId': model_id, 'trainingFeatureMappings': training_feature_mappings}, parse_type=ModelMonitor)

    def rerun_model_monitor(self, model_monitor_id: str) -> ModelMonitor:
        """Reruns the specified model monitor.

        Args:
            model_monitor_id (str): The model monitor to rerun.

        Returns:
            ModelMonitor: The model monitor that is being rerun."""
        return self._call_api('rerunModelMonitor', 'POST', query_params={}, body={'modelMonitorId': model_monitor_id}, parse_type=ModelMonitor)

    def rename_model_monitor(self, model_monitor_id: str, name: str):
        """Renames a model monitor

        Args:
            model_monitor_id (str): The ID of the model monitor to rename
            name (str): The name to apply to the model monitor"""
        return self._call_api('renameModelMonitor', 'PATCH', query_params={}, body={'modelMonitorId': model_monitor_id, 'name': name})

    def delete_model_monitor(self, model_monitor_id: str):
        """Deletes the specified model monitor and all its versions.

        Args:
            model_monitor_id (str): The ID of the model monitor to delete."""
        return self._call_api('deleteModelMonitor', 'DELETE', query_params={'modelMonitorId': model_monitor_id})

    def delete_model_monitor_version(self, model_monitor_version: str):
        """Deletes the specified model monitor version.

        Args:
            model_monitor_version (str): The ID of the model monitor version to delete."""
        return self._call_api('deleteModelMonitorVersion', 'DELETE', query_params={'modelMonitorVersion': model_monitor_version})

    def create_deployment(self, name: str = None, model_id: str = None, feature_group_id: str = None, project_id: str = None, description: str = None, calls_per_second: int = None, auto_deploy: bool = True, start: bool = True, enable_batch_streaming_updates: bool = False) -> Deployment:
        """Creates a deployment with the specified name and description for the specified model or feature group.

        A Deployment makes the trained model or feature group available for prediction requests.


        Args:
            name (str): The name of the deployment.
            model_id (str): The unique ID associated with the model.
            feature_group_id (str): The unique ID associated with a feature group.
            project_id (str): The unique ID associated with a project.
            description (str): The description for the deployment.
            calls_per_second (int): The number of calls per second the deployment could handle.
            auto_deploy (bool): Flag to enable the automatic deployment when a new Model Version finishes training.
            start (bool): 
            enable_batch_streaming_updates (bool): Flag to enable marking the feature group deployment to have a background process cache streamed in rows for quicker lookup

        Returns:
            Deployment: The new model or feature group deployment."""
        return self._call_api('createDeployment', 'POST', query_params={}, body={'name': name, 'modelId': model_id, 'featureGroupId': feature_group_id, 'projectId': project_id, 'description': description, 'callsPerSecond': calls_per_second, 'autoDeploy': auto_deploy, 'start': start, 'enableBatchStreamingUpdates': enable_batch_streaming_updates}, parse_type=Deployment)

    def create_deployment_token(self, project_id: str) -> DeploymentAuthToken:
        """Creates a deployment token for the specified project.

        Deployment tokens are used to authenticate requests to the prediction APIs and are scoped on the project level.


        Args:
            project_id (str): The unique ID associated with the project.

        Returns:
            DeploymentAuthToken: The deployment token."""
        return self._call_api('createDeploymentToken', 'POST', query_params={}, body={'projectId': project_id}, parse_type=DeploymentAuthToken)

    def update_deployment(self, deployment_id: str, description: str = None):
        """Updates a deployment's description.

        Args:
            deployment_id (str): The deployment to update.
            description (str): The new deployment description."""
        return self._call_api('updateDeployment', 'PATCH', query_params={'deploymentId': deployment_id}, body={'description': description})

    def rename_deployment(self, deployment_id: str, name: str):
        """Updates a deployment's name and/or description.

        Args:
            deployment_id (str): The deployment to update.
            name (str): The new deployment name."""
        return self._call_api('renameDeployment', 'PATCH', query_params={'deploymentId': deployment_id}, body={'name': name})

    def set_auto_deployment(self, deployment_id: str, enable: bool = None):
        """Enable/Disable auto deployment for the specified deployment.

        When a model is scheduled to retrain, deployments with this enabled will be marked to automatically promote the new model
        version. After the newly trained model completes, a check on its metrics in comparison to the currently deployed model version
        will be performed. If the metrics are comparable or better, the newly trained model version is automatically promoted. If not,
        it will be marked as a failed model version promotion with an error indicating poor metrics performance.


        Args:
            deployment_id (str): The unique ID associated with the deployment
            enable (bool): Enable/disable the autoDeploy property of the Deployment."""
        return self._call_api('setAutoDeployment', 'POST', query_params={'deploymentId': deployment_id}, body={'enable': enable})

    def set_deployment_model_version(self, deployment_id: str, model_version: str):
        """Promotes a Model Version to be served in the Deployment

        Args:
            deployment_id (str): The unique ID for the Deployment
            model_version (str): The unique ID for the Model Version"""
        return self._call_api('setDeploymentModelVersion', 'PATCH', query_params={'deploymentId': deployment_id}, body={'modelVersion': model_version})

    def set_deployment_feature_group_version(self, deployment_id: str, feature_group_version: str):
        """Promotes a Feature Group Version to be served in the Deployment

        Args:
            deployment_id (str): The unique ID for the Deployment
            feature_group_version (str): The unique ID for the Feature Group Version"""
        return self._call_api('setDeploymentFeatureGroupVersion', 'PATCH', query_params={'deploymentId': deployment_id}, body={'featureGroupVersion': feature_group_version})

    def start_deployment(self, deployment_id: str):
        """Restarts the specified deployment that was previously suspended.

        Args:
            deployment_id (str): The unique ID associated with the deployment."""
        return self._call_api('startDeployment', 'POST', query_params={'deploymentId': deployment_id}, body={})

    def stop_deployment(self, deployment_id: str):
        """Stops the specified deployment.

        Args:
            deployment_id (str): The Deployment ID"""
        return self._call_api('stopDeployment', 'POST', query_params={'deploymentId': deployment_id}, body={})

    def delete_deployment(self, deployment_id: str):
        """Deletes the specified deployment. The deployment's models will not be affected. Note that the deployments are not recoverable after they are deleted.

        Args:
            deployment_id (str): The ID of the deployment to delete."""
        return self._call_api('deleteDeployment', 'DELETE', query_params={'deploymentId': deployment_id})

    def delete_deployment_token(self, deployment_token: str):
        """Deletes the specified deployment token.

        Args:
            deployment_token (str): The deployment token to delete."""
        return self._call_api('deleteDeploymentToken', 'DELETE', query_params={'deploymentToken': deployment_token})

    def set_deployment_feature_group_export_file_connector_output(self, deployment_id: str, file_format: str = None, output_location: str = None):
        """Sets the export output for the Feature Group Deployment to be a file connector.

        Args:
            deployment_id (str): The deployment for which the export type is set
            file_format (str): 
            output_location (str): the file connector (cloud) location of where to export"""
        return self._call_api('setDeploymentFeatureGroupExportFileConnectorOutput', 'POST', query_params={'deploymentId': deployment_id}, body={'fileFormat': file_format, 'outputLocation': output_location})

    def set_deployment_feature_group_export_database_connector_output(self, deployment_id: str, database_connector_id: str, object_name: str, write_mode: str, database_feature_mapping: dict, id_column: str = None, additional_id_columns: list = None):
        """Sets the export output for the Feature Group Deployment to be a Database connector.

        Args:
            deployment_id (str): The deployment for which the export type is set
            database_connector_id (str): The database connector ID used
            object_name (str): The database connector's object to write to
            write_mode (str): UPSERT or INSERT for writing to the database connector
            database_feature_mapping (dict): The column/feature pairs mapping the features to the database columns
            id_column (str): The id column to use as the upsert key
            additional_id_columns (list): For database connectors which support it, additional ID columns to use as a complex key for upserting"""
        return self._call_api('setDeploymentFeatureGroupExportDatabaseConnectorOutput', 'POST', query_params={'deploymentId': deployment_id}, body={'databaseConnectorId': database_connector_id, 'objectName': object_name, 'writeMode': write_mode, 'databaseFeatureMapping': database_feature_mapping, 'idColumn': id_column, 'additionalIdColumns': additional_id_columns})

    def remove_deployment_feature_group_export_output(self, deployment_id: str):
        """Removes the export type that is set for the Feature Group Deployment

        Args:
            deployment_id (str): The deployment for which the export type is set"""
        return self._call_api('removeDeploymentFeatureGroupExportOutput', 'POST', query_params={'deploymentId': deployment_id}, body={})

    def create_refresh_policy(self, name: str, cron: str, refresh_type: str, project_id: str = None, dataset_ids: list = [], model_ids: list = [], deployment_ids: list = [], batch_prediction_ids: list = [], prediction_metric_ids: list = []) -> RefreshPolicy:
        """Creates a refresh policy with a particular cron pattern and refresh type.

        A refresh policy allows for the scheduling of a particular set of actions at regular intervals. This can be useful for periodically updated data which needs to be re-imported into the project for re-training.


        Args:
            name (str): The name for the refresh policy
            cron (str): A cron-like string specifying the frequency of a refresh policy
            refresh_type (str): The Refresh Type is used to determine what is being refreshed, whether its a single dataset, or dataset and a model, or more.
            project_id (str): Optionally, a Project ID can be specified so that all datasets, models and deployments are captured at the instant this policy was created
            dataset_ids (list): Comma separated list of Dataset IDs
            model_ids (list): Comma separated list of Model IDs
            deployment_ids (list): Comma separated list of Deployment IDs
            batch_prediction_ids (list): Comma separated list of Batch Predictions
            prediction_metric_ids (list): Comma separated list of Prediction Metrics

        Returns:
            RefreshPolicy: The refresh policy created"""
        return self._call_api('createRefreshPolicy', 'POST', query_params={}, body={'name': name, 'cron': cron, 'refreshType': refresh_type, 'projectId': project_id, 'datasetIds': dataset_ids, 'modelIds': model_ids, 'deploymentIds': deployment_ids, 'batchPredictionIds': batch_prediction_ids, 'predictionMetricIds': prediction_metric_ids}, parse_type=RefreshPolicy)

    def delete_refresh_policy(self, refresh_policy_id: str):
        """Delete a refresh policy

        Args:
            refresh_policy_id (str): The unique ID associated with this refresh policy"""
        return self._call_api('deleteRefreshPolicy', 'DELETE', query_params={'refreshPolicyId': refresh_policy_id})

    def pause_refresh_policy(self, refresh_policy_id: str):
        """Pauses a refresh policy

        Args:
            refresh_policy_id (str): The unique ID associated with this refresh policy"""
        return self._call_api('pauseRefreshPolicy', 'POST', query_params={}, body={'refreshPolicyId': refresh_policy_id})

    def resume_refresh_policy(self, refresh_policy_id: str):
        """Resumes a refresh policy

        Args:
            refresh_policy_id (str): The unique ID associated with this refresh policy"""
        return self._call_api('resumeRefreshPolicy', 'POST', query_params={}, body={'refreshPolicyId': refresh_policy_id})

    def run_refresh_policy(self, refresh_policy_id: str):
        """Force a run of the refresh policy.

        Args:
            refresh_policy_id (str): The unique ID associated with this refresh policy"""
        return self._call_api('runRefreshPolicy', 'POST', query_params={}, body={'refreshPolicyId': refresh_policy_id})

    def update_refresh_policy(self, refresh_policy_id: str, name: str = None, cron: str = None) -> RefreshPolicy:
        """Update the name or cron string of a  refresh policy

        Args:
            refresh_policy_id (str): The unique ID associated with this refresh policy
            name (str): Optional, specify to update the name of the refresh policy
            cron (str): Optional, specify to update the cron string describing the schedule from the refresh policy

        Returns:
            RefreshPolicy: The updated refresh policy"""
        return self._call_api('updateRefreshPolicy', 'POST', query_params={}, body={'refreshPolicyId': refresh_policy_id, 'name': name, 'cron': cron}, parse_type=RefreshPolicy)

    def lookup_features(self, deployment_token: str, deployment_id: str, query_data: dict = {}, limit_results: int = None, result_columns: list = None) -> Dict:
        """Returns the feature group deployed in the feature store project.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where 'Key' will be the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and 'Value' will be the unique value of the same entity.
            limit_results (int): If present, will limit the number of results to the value provided.
            result_columns (list): If present, will limit the columns present in each result to the columns specified in this list"""
        return self._call_api('lookupFeatures', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'limitResults': limit_results, 'resultColumns': result_columns}, server_override=self.default_prediction_url)

    def predict(self, deployment_token: str, deployment_id: str, query_data: dict = {}) -> Dict:
        """Returns a prediction for Predictive Modeling

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where 'Key' will be the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and 'Value' will be the unique value of the same entity."""
        return self._call_api('predict', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def predict_multiple(self, deployment_token: str, deployment_id: str, query_data: list = {}) -> Dict:
        """Returns a list of predictions for Predictive Modeling

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (list): This will be a list of dictionaries where 'Key' will be the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and 'Value' will be the unique value of the same entity."""
        return self._call_api('predictMultiple', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def predict_from_datasets(self, deployment_token: str, deployment_id: str, query_data: dict = {}) -> Dict:
        """Returns a list of predictions for Predictive Modeling

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where 'Key' will be the source dataset name and 'Value' will be a list of records corresponding to the dataset rows"""
        return self._call_api('predictFromDatasets', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def predict_lead(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns the probability of a user to be a lead on the basis of his/her interaction with the service/product and user's own attributes (e.g. income, assets, credit score, etc.). Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'user_id' mapped to mapping 'LEAD_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary containing user attributes and/or user's interaction data with the product/service (e.g. number of click, items in cart, etc.)."""
        return self._call_api('predictLead', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def predict_churn(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns a probability of a user to churn out in response to his/her interactions with the item/product/service. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'churn_result' mapped to mapping 'CHURNED_YN' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where 'Key' will be the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and 'Value' will be the unique value of the same entity."""
        return self._call_api('predictChurn', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def predict_takeover(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns a probability for each class label associated with the types of fraud or a 'yes' or 'no' type label for the possibility of fraud. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'account_name' mapped to mapping 'ACCOUNT_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary containing account activity characteristics (e.g. login id, login duration, login type, ip address, etc.)."""
        return self._call_api('predictTakeover', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def predict_fraud(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns a probability of a transaction performed under a specific account as being a fraud or not. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'account_number' mapped to the mapping 'ACCOUNT_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary containing transaction attributes (e.g. credit card type, transaction location, transaction amount, etc.)."""
        return self._call_api('predictFraud', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def predict_class(self, deployment_token: str, deployment_id: str, query_data: dict = {}, threshold: float = None, threshold_class: str = None, thresholds: list = None, explain_predictions: bool = False, fixed_features: list = None, nested: str = None, explainer_type: str = None) -> Dict:
        """Returns a classification prediction

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where 'Key' will be the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and 'Value' will be the unique value of the same entity.
            threshold (float): float value that is applied on the popular class label.
            threshold_class (str): label upon which the threshold is added (Binary labels only)
            thresholds (list): maps labels to thresholds (Multi label classification only). Defaults to F1 optimal threshold if computed for the given class, else uses 0.5
            explain_predictions (bool): If true, returns the SHAP explanations for all input features.
            fixed_features (list): Set of input features to treat as constant for explanations.
            nested (str): If specified generates prediction delta for each index of the specified nested feature.
            explainer_type (str): """
        return self._call_api('predictClass', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'threshold': threshold, 'thresholdClass': threshold_class, 'thresholds': thresholds, 'explainPredictions': explain_predictions, 'fixedFeatures': fixed_features, 'nested': nested, 'explainerType': explainer_type}, server_override=self.default_prediction_url)

    def predict_target(self, deployment_token: str, deployment_id: str, query_data: dict = {}, explain_predictions: bool = False, fixed_features: list = None, nested: str = None, explainer_type: str = None) -> Dict:
        """Returns a prediction from a classification or regression model. Optionally, includes explanations.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where 'Key' will be the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and 'Value' will be the unique value of the same entity.
            explain_predictions (bool): If true, returns the SHAP explanations for all input features.
            fixed_features (list): Set of input features to treat as constant for explanations.
            nested (str): If specified generates prediction delta for each index of the specified nested feature.
            explainer_type (str): """
        return self._call_api('predictTarget', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'explainPredictions': explain_predictions, 'fixedFeatures': fixed_features, 'nested': nested, 'explainerType': explainer_type}, server_override=self.default_prediction_url)

    def get_anomalies(self, deployment_token: str, deployment_id: str, threshold: float = None, histogram: bool = False) -> io.BytesIO:
        """Returns a list of anomalies from the training dataset

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            threshold (float): The threshold score of what is an anomaly. Valid values are between 0.8 and 0.99.
            histogram (bool): If True, will return a histogram of the distribution of all points"""
        return self._call_api('getAnomalies', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'threshold': threshold, 'histogram': histogram}, server_override=self.default_prediction_url)

    def is_anomaly(self, deployment_token: str, deployment_id: str, query_data: dict = None) -> Dict:
        """Returns a list of anomaly attributes based on login information for a specified account. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'account_name' mapped to mapping 'ACCOUNT_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): The input data for the prediction."""
        return self._call_api('isAnomaly', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def get_forecast(self, deployment_token: str, deployment_id: str, query_data: dict, future_data: dict = None, num_predictions: int = None, prediction_start: str = None) -> Dict:
        """Returns a list of forecasts for a given entity under the specified project deployment. Note that the inputs to the deployed model will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'holiday_yn' mapped to mapping 'FUTURE' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where 'Key' will be the column name (e.g. a column with name 'store_id' in your dataset) mapped to the column mapping ITEM_ID that uniquely identifies the entity against which forecasting is performed and 'Value' will be the unique value of the same entity.
            future_data (dict): This will be a dictionary of values known ahead of time that are relevant for forecasting (e.g. State Holidays, National Holidays, etc.). The key and the value both will be of type 'String'. For example future data entered for a Store may be {"Holiday":"No", "Promo":"Yes"}.
            num_predictions (int): The number of timestamps to predict in the future.
            prediction_start (str): The start date for predictions (e.g., "2015-08-01T00:00:00" as input for mid-night of 2015-08-01)."""
        return self._call_api('getForecast', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'futureData': future_data, 'numPredictions': num_predictions, 'predictionStart': prediction_start}, server_override=self.default_prediction_url)

    def get_k_nearest(self, deployment_token: str, deployment_id: str, vector: list, k: int = None, distance: str = None, include_score: bool = False) -> Dict:
        """Returns the k nearest neighbors for the provided embedding vector.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            vector (list): Input vector to perform the k nearest neighbors with.
            k (int): Overrideable number of items to return
            distance (str): Specify the distance function to use when finding nearest neighbors
            include_score (bool): If True, will return the score alongside the resulting embedding value"""
        return self._call_api('getKNearest', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'vector': vector, 'k': k, 'distance': distance, 'includeScore': include_score}, server_override=self.default_prediction_url)

    def get_multiple_k_nearest(self, deployment_token: str, deployment_id: str, queries: list):
        """Returns the k nearest neighbors for the queries provided

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            queries (list): List of Mappings of format {"catalogId": "cat0", "vectors": [...], "k": 20, "distance": "euclidean"}. See getKNearest for additional information about the supported parameters"""
        return self._call_api('getMultipleKNearest', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queries': queries}, server_override=self.default_prediction_url)

    def get_labels(self, deployment_token: str, deployment_id: str, query_data: dict, threshold: None = None) -> Dict:
        """Returns a list of scored labels from

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): Dictionary where key is "Content" and value is the text from which entities are to be extracted.
            threshold (None): Deprecated"""
        return self._call_api('getLabels', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'threshold': threshold}, server_override=self.default_prediction_url)

    def get_recommendations(self, deployment_token: str, deployment_id: str, query_data: dict, num_items: int = 50, page: int = 1, exclude_item_ids: list = [], score_field: str = '', scaling_factors: list = [], restrict_items: list = [], exclude_items: list = [], explore_fraction: float = 0.0) -> Dict:
        """Returns a list of recommendations for a given user under the specified project deployment. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'time' mapped to mapping 'TIMESTAMP' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where 'Key' will be the column name (e.g. a column with name 'user_name' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the user against which recommendations are made and 'Value' will be the unique value of the same item. For example, if you have the column name 'user_name' mapped to the column mapping 'USER_ID', then the query must have the exact same column name (user_name) as key and the name of the user (John Doe) as value.
            num_items (int): The number of items to recommend on one page. By default, it is set to 50 items per page.
            page (int): The page number to be displayed. For example, let's say that the num_items is set to 10 with the total recommendations list size of 50 recommended items, then an input value of 2 in the 'page' variable will display a list of items that rank from 11th to 20th.
            exclude_item_ids (list): [DEPRECATED]
            score_field (str): The relative item scores are returned in a separate field named with the same name as the key (score_field) for this argument.
            scaling_factors (list): It allows you to bias the model towards certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1"], "factor": 1.1}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" in reference to which the model recommendations need to be biased; and the key, "factor" takes the factor by which the item scores are adjusted.  Let's take an example where the input to scaling_factors is [{"column": "VehicleType", "values": ["SUV", "Sedan"], "factor": 1.4}]. After we apply the model to get item probabilities, for every SUV and Sedan in the list, we will multiply the respective probability by 1.1 before sorting. This is particularly useful if there's a type of item that might be less popular but you want to promote it or there's an item that always comes up and you want to demote it.
            restrict_items (list): It allows you to restrict the recommendations to certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1", "value3", ...]}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1", "value3", ...]" to which to restrict the recommendations to. Let's take an example where the input to restrict_items is [{"column": "VehicleType", "values": ["SUV", "Sedan"]}]. This input will restrict the recommendations to SUVs and Sedans. This type of restrition is particularly useful if there's a list of items that you know is of use in some particular scenario and you want to restrict the recommendations only to that list.
            exclude_items (list): It allows you to exclude certain items from the list of recommendations. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1", ...]}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" to exclude from the recommendations. Let's take an example where the input to exclude_items is [{"column": "VehicleType", "values": ["SUV", "Sedan"]}]. The resulting recommendation list will exclude all SUVs and Sedans. This is particularly useful if there's a list of items that you know is of no use in some particular scenario and you don't want to show those items present in that list.
            explore_fraction (float): The fraction of recommendations that is to be new items."""
        return self._call_api('getRecommendations', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'numItems': num_items, 'page': page, 'excludeItemIds': exclude_item_ids, 'scoreField': score_field, 'scalingFactors': scaling_factors, 'restrictItems': restrict_items, 'excludeItems': exclude_items, 'exploreFraction': explore_fraction}, server_override=self.default_prediction_url)

    def get_personalized_ranking(self, deployment_token: str, deployment_id: str, query_data: dict, preserve_ranks: list = [], preserve_unknown_items: bool = False, scaling_factors: list = []) -> Dict:
        """Returns a list of items with personalized promotions on them for a given user under the specified project deployment. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'item_code' mapped to mapping 'ITEM_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary with two key-value pairs. The first pair represents a 'Key' where the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID uniquely identifies the user against whom a prediction is made and a 'Value' which is the identifier value for that user. The second pair will have a 'Key' which will be the name of the column name (e.g. movie_name) mapped to ITEM_ID (unique item identifier) and a 'Value' which will be a list of identifiers that uniquely identifies those items.
            preserve_ranks (list): List of dictionaries of format {"column": "col0", "values": ["value0, value1"]}, where the ranks of items in query_data is preserved for all the items in "col0" with values, "value0" and "value1". This option is useful when the desired items are being recommended in the desired order and the ranks for those items need to be kept unchanged during recommendation generation.
            preserve_unknown_items (bool): If true, any items that are unknown to the model, will not be reranked, and the original position in the query will be preserved.
            scaling_factors (list): It allows you to bias the model towards certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1"], "factor": 1.1}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" in reference to which the model recommendations need to be biased; and the key, "factor" takes the factor by which the item scores are adjusted.  Let's take an example where the input to scaling_factors is [{"column": "VehicleType", "values": ["SUV", "Sedan"], "factor": 1.4}]. After we apply the model to get item probabilities, for every SUV and Sedan in the list, we will multiply the respective probability by 1.1 before sorting. This is particularly useful if there's a type of item that might be less popular but you want to promote it or there's an item that always comes up and you want to demote it."""
        return self._call_api('getPersonalizedRanking', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'preserveRanks': preserve_ranks, 'preserveUnknownItems': preserve_unknown_items, 'scalingFactors': scaling_factors}, server_override=self.default_prediction_url)

    def get_ranked_items(self, deployment_token: str, deployment_id: str, query_data: dict, preserve_ranks: list = [], preserve_unknown_items: bool = False, scaling_factors: list = []) -> Dict:
        """Returns a list of re-ranked items for a selected user when a list of items is required to be reranked according to the user's preferences. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'item_code' mapped to mapping 'ITEM_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary with two key-value pairs. The first pair represents a 'Key' where the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID uniquely identifies the user against whom a prediction is made and a 'Value' which is the identifier value for that user. The second pair will have a 'Key' which will be the name of the column name (e.g. movie_name) mapped to ITEM_ID (unique item identifier) and a 'Value' which will be a list of identifiers that uniquely identifies those items.
            preserve_ranks (list): List of dictionaries of format {"column": "col0", "values": ["value0, value1"]}, where the ranks of items in query_data is preserved for all the items in "col0" with values, "value0" and "value1". This option is useful when the desired items are being recommended in the desired order and the ranks for those items need to be kept unchanged during recommendation generation.
            preserve_unknown_items (bool): If true, any items that are unknown to the model, will not be reranked, and the original position in the query will be preserved.
            scaling_factors (list): It allows you to bias the model towards certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1"], "factor": 1.1}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" in reference to which the model recommendations need to be biased; and the key, "factor" takes the factor by which the item scores are adjusted.  Let's take an example where the input to scaling_factors is [{"column": "VehicleType", "values": ["SUV", "Sedan"], "factor": 1.4}]. After we apply the model to get item probabilities, for every SUV and Sedan in the list, we will multiply the respective probability by 1.1 before sorting. This is particularly useful if there's a type of item that might be less popular but you want to promote it or there's an item that always comes up and you want to demote it."""
        return self._call_api('getRankedItems', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'preserveRanks': preserve_ranks, 'preserveUnknownItems': preserve_unknown_items, 'scalingFactors': scaling_factors}, server_override=self.default_prediction_url)

    def get_related_items(self, deployment_token: str, deployment_id: str, query_data: dict, num_items: int = 50, page: int = 1, scaling_factors: list = [], restrict_items: list = [], exclude_items: list = []) -> Dict:
        """Returns a list of related items for a given item under the specified project deployment. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'item_code' mapped to mapping 'ITEM_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where 'Key' will be the column name (e.g. a column with name 'user_name' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the user against which related items are determined and 'Value' will be the unique value of the same item. For example, if you have the column name 'user_name' mapped to the column mapping 'USER_ID', then the query must have the exact same column name (user_name) as key and the name of the user (John Doe) as value.
            num_items (int): The number of items to recommend on one page. By default, it is set to 50 items per page.
            page (int): The page number to be displayed. For example, let's say that the num_items is set to 10 with the total recommendations list size of 50 recommended items, then an input value of 2 in the 'page' variable will display a list of items that rank from 11th to 20th.
            scaling_factors (list): It allows you to bias the model towards certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1"], "factor": 1.1}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" in reference to which the model recommendations need to be biased; and the key, "factor" takes the factor by which the item scores are adjusted.  Let's take an example where the input to scaling_factors is [{"column": "VehicleType", "values": ["SUV", "Sedan"], "factor": 1.4}]. After we apply the model to get item probabilities, for every SUV and Sedan in the list, we will multiply the respective probability by 1.1 before sorting. This is particularly useful if there's a type of item that might be less popular but you want to promote it or there's an item that always comes up and you want to demote it.
            restrict_items (list): It allows you to restrict the recommendations to certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1", "value3", ...]}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1", "value3", ...]" to which to restrict the recommendations to. Let's take an example where the input to restrict_items is [{"column": "VehicleType", "values": ["SUV", "Sedan"]}]. This input will restrict the recommendations to SUVs and Sedans. This type of restrition is particularly useful if there's a list of items that you know is of use in some particular scenario and you want to restrict the recommendations only to that list.
            exclude_items (list): It allows you to exclude certain items from the list of recommendations. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1", ...]}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" to exclude from the recommendations. Let's take an example where the input to exclude_items is [{"column": "VehicleType", "values": ["SUV", "Sedan"]}]. The resulting recommendation list will exclude all SUVs and Sedans. This is particularly useful if there's a list of items that you know is of no use in some particular scenario and you don't want to show those items present in that list."""
        return self._call_api('getRelatedItems', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'numItems': num_items, 'page': page, 'scalingFactors': scaling_factors, 'restrictItems': restrict_items, 'excludeItems': exclude_items}, server_override=self.default_prediction_url)

    def get_feature_group_rows(self, deployment_token: str, deployment_id: str, query_data: dict):
        """

        Args:
            deployment_token (str): 
            deployment_id (str): 
            query_data (dict): """
        return self._call_api('getFeatureGroupRows', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def get_search_results(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """TODO

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): Dictionary where key is "Content" and value is the text from which entities are to be extracted."""
        return self._call_api('getSearchResults', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def get_sentiment(self, deployment_token: str, deployment_id: str, document: str) -> Dict:
        """TODO

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            document (str): # TODO"""
        return self._call_api('getSentiment', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'document': document}, server_override=self.default_prediction_url)

    def get_entailment(self, deployment_token: str, deployment_id: str, document: str) -> Dict:
        """TODO

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            document (str): # TODO"""
        return self._call_api('getEntailment', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'document': document}, server_override=self.default_prediction_url)

    def get_classification(self, deployment_token: str, deployment_id: str, document: str) -> Dict:
        """TODO

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            document (str): # TODO"""
        return self._call_api('getClassification', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'document': document}, server_override=self.default_prediction_url)

    def get_summary(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns a json of the predicted summary for the given document. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'text' mapped to mapping 'DOCUMENT' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): Raw Data dictionary containing the required document data - must have a key document corresponding to a DOCUMENT type text as value."""
        return self._call_api('getSummary', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def predict_language(self, deployment_token: str, deployment_id: str, query_data: str) -> Dict:
        """TODO

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (str): # TODO"""
        return self._call_api('predictLanguage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def get_assignments(self, deployment_token: str, deployment_id: str, query_data: dict, forced_assignments: dict = None) -> Dict:
        """Get all positive assignments that match a query.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): specifies the set of assignments being requested.
            forced_assignments (dict): set of assignments to force and resolve before returning query results."""
        return self._call_api('getAssignments', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'forcedAssignments': forced_assignments}, server_override=self.default_prediction_url)

    def check_constraints(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Check for any constraints violated by the overrides.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): assignment overrides to the solution."""
        return self._call_api('checkConstraints', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def create_prediction_metric(self, feature_group_id: str, prediction_metric_config: dict, project_id: str = None) -> PredictionMetric:
        """Create a prediction metric job description for the given prediction and actual-labels data.

        Args:
            feature_group_id (str): The feature group to use as input to the prediction metrics.
            prediction_metric_config (dict): Specification for prediction metric to run in this job.
            project_id (str): Project to use for the prediction metrics. Defaults to the project for the input feature_group, if the feature_group has exactly one project.

        Returns:
            PredictionMetric: The Prediction Metric job description."""
        return self._call_api('createPredictionMetric', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'predictionMetricConfig': prediction_metric_config, 'projectId': project_id}, parse_type=PredictionMetric)

    def describe_prediction_metric(self, prediction_metric_id: str, should_include_latest_version_description: bool = True) -> PredictionMetric:
        """Describe a Prediction Metric.

        Args:
            prediction_metric_id (str): The unique ID associated with the prediction metric.
            should_include_latest_version_description (bool): include the description of the latest prediction metric version

        Returns:
            PredictionMetric: The prediction metric object."""
        return self._call_api('describePredictionMetric', 'POST', query_params={}, body={'predictionMetricId': prediction_metric_id, 'shouldIncludeLatestVersionDescription': should_include_latest_version_description}, parse_type=PredictionMetric)

    def delete_prediction_metric(self, prediction_metric_id: str):
        """Removes an existing PredictionMetric.

        Args:
            prediction_metric_id (str): The unique ID associated with the prediction metric."""
        return self._call_api('deletePredictionMetric', 'DELETE', query_params={'predictionMetricId': prediction_metric_id})

    def run_prediction_metric(self, prediction_metric_id: str) -> PredictionMetricVersion:
        """Creates a new prediction metrics job run for the given prediction metric job description, and starts that job.

        Configures and starts the computations running to compute the prediciton metric.


        Args:
            prediction_metric_id (str): The prediction metric job description to apply for configuring a prediction metric job.

        Returns:
            PredictionMetricVersion: A prediction metric version. For more information, please refer to the details on the object (below)."""
        return self._call_api('runPredictionMetric', 'POST', query_params={}, body={'predictionMetricId': prediction_metric_id}, parse_type=PredictionMetricVersion)

    def delete_prediction_metric_version(self, prediction_metric_version: str):
        """Removes an existing prediction metric version.

        Args:
            prediction_metric_version (str): """
        return self._call_api('deletePredictionMetricVersion', 'DELETE', query_params={'predictionMetricVersion': prediction_metric_version})

    def list_prediction_metric_versions(self, prediction_metric_id: str, limit: int = 100, start_after_id: str = None) -> List[PredictionMetricVersion]:
        """List the prediction metric versions for a prediction metric.

        Args:
            prediction_metric_id (str): The prediction metric whose instances will be listed.
            limit (int): The the number of prediction metric instances to be retrieved.
            start_after_id (str): An offset parameter to exclude all prediction metric versions till the specified prediction metric ID.

        Returns:
            PredictionMetricVersion: The prediction metric instances for this prediction metric."""
        return self._call_api('listPredictionMetricVersions', 'POST', query_params={}, body={'predictionMetricId': prediction_metric_id, 'limit': limit, 'startAfterId': start_after_id}, parse_type=PredictionMetricVersion)

    def create_batch_prediction(self, deployment_id: str, table_name: str = None, name: str = None, global_prediction_args: dict = None, explanations: bool = False, output_format: str = None, output_location: str = None, database_connector_id: str = None, database_output_config: dict = None, refresh_schedule: str = None, csv_input_prefix: str = None, csv_prediction_prefix: str = None, csv_explanations_prefix: str = None, output_includes_metadata: bool = None, result_input_columns: list = None) -> BatchPrediction:
        """Creates a batch prediction job description for the given deployment.

        Args:
            deployment_id (str): The unique identifier to a deployment.
            table_name (str): If specified, the name of the feature group table to write the results of the batch prediction. Can only be specified iff outputLocation and databaseConnectorId are not specified. If tableName is specified, the outputType will be enforced as CSV
            name (str): The name of batch prediction job.
            global_prediction_args (dict): Argument(s) to pass on every prediction call.
            explanations (bool): If true, will provide SHAP Explanations for each prediction, if supported by the use case.
            output_format (str): If specified, sets the format of the batch prediction output (CSV or JSON)
            output_location (str): If specified, the location to write the prediction results. Otherwise, results will be stored in Abacus.AI.
            database_connector_id (str): The unique identifier of an Database Connection to write predictions to. Cannot be specified in conjunction with outputLocation.
            database_output_config (dict): A key-value pair of columns/values to write to the database connector. Only available if databaseConnectorId is specified.
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically run the batch prediction.
            csv_input_prefix (str): A prefix to prepend to the input columns, only applies when output format is CSV
            csv_prediction_prefix (str): A prefix to prepend to the prediction columns, only applies when output format is CSV
            csv_explanations_prefix (str): A prefix to prepend to the explanation columns, only applies when output format is CSV
            output_includes_metadata (bool): If true, output will contain columns including prediction start time, batch prediction version, and model version
            result_input_columns (list): If present, will limit result files or feature groups to only include columns present in this list

        Returns:
            BatchPrediction: The batch prediction description."""
        return self._call_api('createBatchPrediction', 'POST', query_params={'deploymentId': deployment_id}, body={'tableName': table_name, 'name': name, 'globalPredictionArgs': global_prediction_args, 'explanations': explanations, 'outputFormat': output_format, 'outputLocation': output_location, 'databaseConnectorId': database_connector_id, 'databaseOutputConfig': database_output_config, 'refreshSchedule': refresh_schedule, 'csvInputPrefix': csv_input_prefix, 'csvPredictionPrefix': csv_prediction_prefix, 'csvExplanationsPrefix': csv_explanations_prefix, 'outputIncludesMetadata': output_includes_metadata, 'resultInputColumns': result_input_columns}, parse_type=BatchPrediction)

    def start_batch_prediction(self, batch_prediction_id: str) -> BatchPredictionVersion:
        """Creates a new batch prediction version job for a given batch prediction job description

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction to create a new version of

        Returns:
            BatchPredictionVersion: The batch prediction version started by this method call."""
        return self._call_api('startBatchPrediction', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id}, parse_type=BatchPredictionVersion)

    def update_batch_prediction(self, batch_prediction_id: str, deployment_id: str = None, global_prediction_args: dict = None, explanations: bool = None, output_format: str = None, csv_input_prefix: str = None, csv_prediction_prefix: str = None, csv_explanations_prefix: str = None, output_includes_metadata: bool = None, result_input_columns: list = None) -> BatchPrediction:
        """Updates a batch prediction job description

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction.
            deployment_id (str): The unique identifier to a deployment.
            global_prediction_args (dict): Argument(s) to pass on every prediction call.
            explanations (bool): If true, will provide SHAP Explanations for each prediction, if supported by the use case.
            output_format (str): If specified, sets the format of the batch prediction output (CSV or JSON).
            csv_input_prefix (str): A prefix to prepend to the input columns, only applies when output format is CSV
            csv_prediction_prefix (str): A prefix to prepend to the prediction columns, only applies when output format is CSV
            csv_explanations_prefix (str): A prefix to prepend to the explanation columns, only applies when output format is CSV
            output_includes_metadata (bool): If true, output will contain columns including prediction start time, batch prediction version, and model version
            result_input_columns (list): If present, will limit result files or feature groups to only include columns present in this list

        Returns:
            BatchPrediction: The batch prediction description."""
        return self._call_api('updateBatchPrediction', 'POST', query_params={'deploymentId': deployment_id}, body={'batchPredictionId': batch_prediction_id, 'globalPredictionArgs': global_prediction_args, 'explanations': explanations, 'outputFormat': output_format, 'csvInputPrefix': csv_input_prefix, 'csvPredictionPrefix': csv_prediction_prefix, 'csvExplanationsPrefix': csv_explanations_prefix, 'outputIncludesMetadata': output_includes_metadata, 'resultInputColumns': result_input_columns}, parse_type=BatchPrediction)

    def set_batch_prediction_file_connector_output(self, batch_prediction_id: str, output_format: str = None, output_location: str = None) -> BatchPrediction:
        """Updates the file connector output configuration of the batch prediction

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction.
            output_format (str): If specified, sets the format of the batch prediction output (CSV or JSON).
            output_location (str): If specified, the location to write the prediction results. Otherwise, results will be stored in Abacus.AI.

        Returns:
            BatchPrediction: The batch prediction description."""
        return self._call_api('setBatchPredictionFileConnectorOutput', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id, 'outputFormat': output_format, 'outputLocation': output_location}, parse_type=BatchPrediction)

    def set_batch_prediction_database_connector_output(self, batch_prediction_id: str, database_connector_id: str = None, database_output_config: dict = None) -> BatchPrediction:
        """Updates the database connector output configuration of the batch prediction

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction
            database_connector_id (str): The unique identifier of an Database Connection to write predictions to.
            database_output_config (dict): A key-value pair of columns/values to write to the database connector

        Returns:
            BatchPrediction: The batch prediction description."""
        return self._call_api('setBatchPredictionDatabaseConnectorOutput', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id, 'databaseConnectorId': database_connector_id, 'databaseOutputConfig': database_output_config}, parse_type=BatchPrediction)

    def set_batch_prediction_feature_group_output(self, batch_prediction_id: str, table_name: str) -> BatchPrediction:
        """Creates a feature group and sets it to be the batch prediction output

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction
            table_name (str): The name of the feature group table to create

        Returns:
            BatchPrediction: The batch prediction after the output has been applied"""
        return self._call_api('setBatchPredictionFeatureGroupOutput', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id, 'tableName': table_name}, parse_type=BatchPrediction)

    def set_batch_prediction_output_to_console(self, batch_prediction_id: str) -> BatchPrediction:
        """Sets the batch prediction output to the console, clearing both the file connector and database connector config

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction

        Returns:
            BatchPrediction: The batch prediction description."""
        return self._call_api('setBatchPredictionOutputToConsole', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id}, parse_type=BatchPrediction)

    def set_batch_prediction_dataset(self, batch_prediction_id: str, dataset_type: str, dataset_id: str = None) -> BatchPrediction:
        """[Deprecated] Sets the batch prediction input dataset. Only applicable for legacy dataset-based projects

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction
            dataset_type (str): The dataset type to set
            dataset_id (str): The dataset to set

        Returns:
            BatchPrediction: The batch prediction description."""
        return self._call_api('setBatchPredictionDataset', 'POST', query_params={'datasetId': dataset_id}, body={'batchPredictionId': batch_prediction_id, 'datasetType': dataset_type}, parse_type=BatchPrediction)

    def set_batch_prediction_feature_group(self, batch_prediction_id: str, feature_group_type: str, feature_group_id: str = None) -> BatchPrediction:
        """Sets the batch prediction input feature group.

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction
            feature_group_type (str): The feature group type to set. The feature group type of the feature group. The type is based on the use case under which the feature group is being created. For example, Catalog Attributes can be a feature group type under personalized recommendation use case.
            feature_group_id (str): The feature group to set as input to the batch prediction

        Returns:
            BatchPrediction: The batch prediction description."""
        return self._call_api('setBatchPredictionFeatureGroup', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id, 'featureGroupType': feature_group_type, 'featureGroupId': feature_group_id}, parse_type=BatchPrediction)

    def set_batch_prediction_dataset_remap(self, batch_prediction_id: str, dataset_id_remap: dict) -> BatchPrediction:
        """For the purpose of this batch prediction, will swap out datasets in the input feature groups

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction
            dataset_id_remap (dict): Key/value pairs of dataset_ids to replace during batch predictions

        Returns:
            BatchPrediction: Batch Prediction object"""
        return self._call_api('setBatchPredictionDatasetRemap', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id, 'datasetIdRemap': dataset_id_remap}, parse_type=BatchPrediction)

    def delete_batch_prediction(self, batch_prediction_id: str):
        """Deletes a batch prediction

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction"""
        return self._call_api('deleteBatchPrediction', 'DELETE', query_params={'batchPredictionId': batch_prediction_id})

    def add_user_item_interaction(self, streaming_token: str, dataset_id: str, timestamp: int, user_id: str, item_id: list, event_type: str, additional_attributes: dict):
        """Adds a user-item interaction record (data row) to a streaming dataset.

        Args:
            streaming_token (str): The streaming token for authenticating requests to the dataset.
            dataset_id (str): The streaming dataset to record data to.
            timestamp (int): The unix timestamp of the event.
            user_id (str): The unique identifier for the user.
            item_id (list): The unique identifier for the items
            event_type (str): The event type.
            additional_attributes (dict): Attributes of the user interaction."""
        return self._call_api('addUserItemInteraction', 'POST', query_params={'streamingToken': streaming_token, 'datasetId': dataset_id}, body={'timestamp': timestamp, 'userId': user_id, 'itemId': item_id, 'eventType': event_type, 'additionalAttributes': additional_attributes})

    def upsert_user_attributes(self, streaming_token: str, dataset_id: str, user_id: str, user_attributes: dict):
        """Adds a user attributes record (data row) to a streaming dataset.

        Either the streaming dataset ID or the project ID is required.


        Args:
            streaming_token (str): The streaming token for authenticating requests to the dataset.
            dataset_id (str): The streaming dataset to record data to.
            user_id (str): The unique identifier for the user.
            user_attributes (dict): Attributes of the user interaction."""
        return self._call_api('upsertUserAttributes', 'POST', query_params={'streamingToken': streaming_token, 'datasetId': dataset_id}, body={'userId': user_id, 'userAttributes': user_attributes})

    def upsert_item_attributes(self, streaming_token: str, dataset_id: str, item_id: str, item_attributes: dict):
        """Adds an item attributes record (data row) to a streaming dataset.

        Either the streaming dataset ID or the project ID is required.


        Args:
            streaming_token (str): The streaming token for authenticating requests to the dataset.
            dataset_id (str): The streaming dataset to record data to.
            item_id (str): The unique identifier for the item.
            item_attributes (dict): Attributes of the item interaction."""
        return self._call_api('upsertItemAttributes', 'POST', query_params={'streamingToken': streaming_token, 'datasetId': dataset_id}, body={'itemId': item_id, 'itemAttributes': item_attributes})

    def add_multiple_user_item_interactions(self, streaming_token: str, dataset_id: str, interactions: list):
        """Adds a user-item interaction record (data row) to a streaming dataset.

        Args:
            streaming_token (str): The streaming token for authenticating requests to the dataset.
            dataset_id (str): The streaming dataset to record data to.
            interactions (list): List of interactions, each interaction of format {'userId': userId, 'timestamp': timestamp, 'itemId': itemId, 'eventType': eventType, 'additionalAttributes': {'attribute1': 'abc', 'attribute2': 123}}"""
        return self._call_api('addMultipleUserItemInteractions', 'POST', query_params={'streamingToken': streaming_token, 'datasetId': dataset_id}, body={'interactions': interactions})

    def upsert_multiple_user_attributes(self, streaming_token: str, dataset_id: str, upserts: list):
        """Adds multiple user attributes records (data row) to a streaming dataset.

        The streaming dataset ID is required.


        Args:
            streaming_token (str): The streaming token for authenticating requests to the dataset.
            dataset_id (str): The streaming dataset to record data to.
            upserts (list): List of upserts, each upsert of format {'userId': userId, 'userAttributes': {'attribute1': 'abc', 'attribute2': 123}}."""
        return self._call_api('upsertMultipleUserAttributes', 'POST', query_params={'streamingToken': streaming_token, 'datasetId': dataset_id}, body={'upserts': upserts})

    def upsert_multiple_item_attributes(self, streaming_token: str, dataset_id: str, upserts: list):
        """Adds multiple item attributes records (data row) to a streaming dataset.

        The streaming dataset ID is required.


        Args:
            streaming_token (str): The streaming token for authenticating requests to the dataset.
            dataset_id (str): The streaming dataset to record data to.
            upserts (list): List of upserts, each upsert of format {'itemId': itemId, 'itemAttributes': {'attribute1': 'abc', 'attribute2': 123}}."""
        return self._call_api('upsertMultipleItemAttributes', 'POST', query_params={'streamingToken': streaming_token, 'datasetId': dataset_id}, body={'upserts': upserts})

    def upsert_item_embeddings(self, streaming_token: str, model_id: str, item_id: str, vector: list, catalog_id: str = None):
        """Upserts an embedding vector for an item id for a model_id.

        Args:
            streaming_token (str): The streaming token for authenticating requests to the model.
            model_id (str): The model id to upsert item embeddings to.
            item_id (str): The item id for which its embeddings will be upserted.
            vector (list): The embedding vector.
            catalog_id (str): Optional name to specify which catalog in a model to update."""
        return self._call_api('upsertItemEmbeddings', 'POST', query_params={'streamingToken': streaming_token}, body={'modelId': model_id, 'itemId': item_id, 'vector': vector, 'catalogId': catalog_id})

    def delete_item_embeddings(self, streaming_token: str, model_id: str, item_ids: list, catalog_id: str = None):
        """Deletes knn embeddings for a list of item ids for a model_id.

        Args:
            streaming_token (str): The streaming token for authenticating requests to the model.
            model_id (str): The model id to delete item embeddings from.
            item_ids (list): A list of item ids for which its embeddings will be deleted.
            catalog_id (str): Optional name to specify which catalog in a model to update."""
        return self._call_api('deleteItemEmbeddings', 'POST', query_params={'streamingToken': streaming_token}, body={'modelId': model_id, 'itemIds': item_ids, 'catalogId': catalog_id})

    def upsert_multiple_item_embeddings(self, streaming_token: str, model_id: str, upserts: list, catalog_id: str = None):
        """Upserts a knn embedding for multiple item ids for a model_id.

        Args:
            streaming_token (str): The streaming token for authenticating requests to the model.
            model_id (str): The model id to upsert item embeddings to.
            upserts (list): A list of {'itemId': ..., 'vector': [...]} dicts for each upsert.
            catalog_id (str): Optional name to specify which catalog in a model to update."""
        return self._call_api('upsertMultipleItemEmbeddings', 'POST', query_params={'streamingToken': streaming_token}, body={'modelId': model_id, 'upserts': upserts, 'catalogId': catalog_id})

    def upsert_data(self, feature_group_id: str, streaming_token: str, data: dict):
        """Updates new data into the feature group for a given lookup key recordId if the recordID is found otherwise inserts new data into the feature group.

        Args:
            feature_group_id (str): The Streaming feature group to record data to
            streaming_token (str): The streaming token for authenticating requests
            data (dict): The data to record"""
        return self._call_api('upsertData', 'POST', query_params={'streamingToken': streaming_token}, body={'featureGroupId': feature_group_id, 'data': data})

    def append_data(self, feature_group_id: str, streaming_token: str, data: dict):
        """Appends new data into the feature group for a given lookup key recordId.

        Args:
            feature_group_id (str): The Streaming feature group to record data to
            streaming_token (str): The streaming token for authenticating requests
            data (dict): The data to record"""
        return self._call_api('appendData', 'POST', query_params={'streamingToken': streaming_token}, body={'featureGroupId': feature_group_id, 'data': data})

    def upsert_multiple_data(self, feature_group_id: str, streaming_token: str, data: dict):
        """Updates new data into the feature group for a given lookup key recordId if the recordID is found otherwise inserts new data into the feature group.

        Args:
            feature_group_id (str): The Streaming feature group to record data to
            streaming_token (str): The streaming token for authenticating requests
            data (dict): The data to record, as an array of JSON Objects"""
        return self._call_api('upsertMultipleData', 'POST', query_params={'streamingToken': streaming_token}, body={'featureGroupId': feature_group_id, 'data': data})

    def append_multiple_data(self, feature_group_id: str, streaming_token: str, data: list):
        """Appends new data into the feature group for a given lookup key recordId.

        Args:
            feature_group_id (str): The Streaming feature group to record data to
            streaming_token (str): The streaming token for authenticating requests
            data (list): The data to record, as an array of JSON objects"""
        return self._call_api('appendMultipleData', 'POST', query_params={'streamingToken': streaming_token}, body={'featureGroupId': feature_group_id, 'data': data})

    def create_algorithm(self, name: str, problem_type: str, source_code: str = None, training_data_parameter_names_mapping: dict = None, training_config_parameter_name: str = None, train_function_name: str = None, predict_function_name: str = None, predict_many_function_name: str = None, initialize_function_name: str = None, config_options: dict = None, is_default_enabled: bool = False, project_id: str = None) -> Algorithm:
        """Creates a custome algorithm that's re-usable for model training

        Args:
            name (str): The name to identify the algorithm, only uppercase letters, numbers and underscore allowed
            problem_type (str): The type of the problem this algorithm will work on
            source_code (str): Contents of a valid python source code file. The source code should contain the train/predict/predict_many/initialize functions. A list of allowed import and system libraries for each language is specified in the user functions documentation section.
            training_data_parameter_names_mapping (dict): The mapping from feature group types to training data parameter names in the train function
            training_config_parameter_name (str): The train config parameter name in the train function
            train_function_name (str): Name of the function found in the source code that will be executed to train the model. It is not executed when this function is run.
            predict_function_name (str): Name of the function found in the source code that will be executed run predictions through model. It is not executed when this function is run.
            predict_many_function_name (str): Name of the function found in the source code that will be executed for batch prediction of the model. It is not executed when this function is run.
            initialize_function_name (str): Name of the function found in the source code to initialize the trained model before using it to make predictions using the model
            config_options (dict): Map dataset types and configs to train function parameter names
            is_default_enabled (bool): Whether train with the algorithm by default
            project_id (str): The unique version ID of the project

        Returns:
            Algorithm: The new customer model can be used for training"""
        return self._call_api('createAlgorithm', 'POST', query_params={}, body={'name': name, 'problemType': problem_type, 'sourceCode': source_code, 'trainingDataParameterNamesMapping': training_data_parameter_names_mapping, 'trainingConfigParameterName': training_config_parameter_name, 'trainFunctionName': train_function_name, 'predictFunctionName': predict_function_name, 'predictManyFunctionName': predict_many_function_name, 'initializeFunctionName': initialize_function_name, 'configOptions': config_options, 'isDefaultEnabled': is_default_enabled, 'projectId': project_id}, parse_type=Algorithm)

    def delete_algorithm(self, algorithm: str):
        """Deletes the specified customer algorithm.

        Args:
            algorithm (str): The name of the algorithm to delete."""
        return self._call_api('deleteAlgorithm', 'DELETE', query_params={'algorithm': algorithm})

    def update_algorithm(self, algorithm: str, source_code: str = None, training_data_parameter_names_mapping: dict = None, training_config_parameter_name: str = None, train_function_name: str = None, predict_function_name: str = None, predict_many_function_name: str = None, initialize_function_name: str = None, config_options: dict = None, is_default_enabled: bool = None) -> Algorithm:
        """Update custome algorithm for the given algorithm name. If source_code is provided, also need to provide all the function names in the source_code.

        Args:
            algorithm (str): The name to identify the algorithm, only uppercase letters, numbers and underscore allowed
            source_code (str): Contents of a valid python source code file. The source code should contain the train/predict/predict_many/initialize functions. A list of allowed import and system libraries for each language is specified in the user functions documentation section.
            training_data_parameter_names_mapping (dict): The mapping from feature group types to training data parameter names in the train function
            training_config_parameter_name (str): The train config parameter name in the train function
            train_function_name (str): Name of the function found in the source code that will be executed to train the model. It is not executed when this function is run.
            predict_function_name (str): Name of the function found in the source code that will be executed run predictions through model. It is not executed when this function is run.
            predict_many_function_name (str): Name of the function found in the source code that will be executed for batch prediction of the model. It is not executed when this function is run.
            initialize_function_name (str): Name of the function found in the source code to initialize the trained model before using it to make predictions using the model
            config_options (dict): Map dataset types and configs to train function parameter names
            is_default_enabled (bool): Whether train with the algorithm by default

        Returns:
            Algorithm: The new customer model can be used for training"""
        return self._call_api('updateAlgorithm', 'PATCH', query_params={}, body={'algorithm': algorithm, 'sourceCode': source_code, 'trainingDataParameterNamesMapping': training_data_parameter_names_mapping, 'trainingConfigParameterName': training_config_parameter_name, 'trainFunctionName': train_function_name, 'predictFunctionName': predict_function_name, 'predictManyFunctionName': predict_many_function_name, 'initializeFunctionName': initialize_function_name, 'configOptions': config_options, 'isDefaultEnabled': is_default_enabled}, parse_type=Algorithm)
