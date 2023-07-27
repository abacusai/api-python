import inspect
import io
import json
import logging
import os
import re
import string
import sys
import tarfile
import tempfile
import threading
import time
import warnings
from functools import lru_cache
from typing import Callable, Dict, List, Union

import pandas as pd
import requests
from packaging import version
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from .algorithm import Algorithm
from .annotation_config import AnnotationConfig
from .annotation_entry import AnnotationEntry
from .annotations_status import AnnotationsStatus
from .api_class import (
    ApiClass, ApiEnum, BatchPredictionArgs, DocumentRetrieverConfig,
    FeatureGroupExportConfig, MergeConfig, ParsingConfig, ProblemType,
    SamplingConfig, TrainingConfig
)
from .api_client_utils import (
    INVALID_PANDAS_COLUMN_NAME_CHARACTERS, clean_column_name,
    get_clean_function_source_code, get_object_from_context
)
from .api_endpoint import ApiEndpoint
from .api_key import ApiKey
from .application_connector import ApplicationConnector
from .batch_prediction import BatchPrediction
from .batch_prediction_version import BatchPredictionVersion
from .chat_session import ChatSession
from .custom_loss_function import CustomLossFunction
from .custom_metric import CustomMetric
from .custom_metric_version import CustomMetricVersion
from .custom_train_function_info import CustomTrainFunctionInfo
from .data_metrics import DataMetrics
from .data_prep_logs import DataPrepLogs
from .database_connector import DatabaseConnector
from .dataset import Dataset
from .dataset_column import DatasetColumn
from .dataset_version import DatasetVersion
from .deployment import Deployment
from .deployment_auth_token import DeploymentAuthToken
from .deployment_conversation import DeploymentConversation
from .document_retriever import DocumentRetriever
from .document_retriever_config import DocumentRetrieverConfig
from .document_retriever_lookup_result import DocumentRetrieverLookupResult
from .document_retriever_version import DocumentRetrieverVersion
from .drift_distributions import DriftDistributions
from .eda import Eda
from .eda_collinearity import EdaCollinearity
from .eda_data_consistency import EdaDataConsistency
from .eda_feature_association import EdaFeatureAssociation
from .eda_feature_collinearity import EdaFeatureCollinearity
from .eda_forecasting_analysis import EdaForecastingAnalysis
from .eda_version import EdaVersion
from .execute_feature_group_operation import ExecuteFeatureGroupOperation
from .feature import Feature
from .feature_distribution import FeatureDistribution
from .feature_group import FeatureGroup
from .feature_group_export import FeatureGroupExport
from .feature_group_export_config import FeatureGroupExportConfig
from .feature_group_export_download_url import FeatureGroupExportDownloadUrl
from .feature_group_template import FeatureGroupTemplate
from .feature_group_version import FeatureGroupVersion
from .feature_importance import FeatureImportance
from .file_connector import FileConnector
from .file_connector_instructions import FileConnectorInstructions
from .file_connector_verification import FileConnectorVerification
from .function_logs import FunctionLogs
from .generated_pit_feature_config_option import GeneratedPitFeatureConfigOption
from .graph_dashboard import GraphDashboard
from .inferred_feature_mappings import InferredFeatureMappings
from .llm_input import LlmInput
from .llm_parameters import LlmParameters
from .llm_response import LlmResponse
from .model import Model
from .model_artifacts_export import ModelArtifactsExport
from .model_metrics import ModelMetrics
from .model_monitor import ModelMonitor
from .model_monitor_org_summary import ModelMonitorOrgSummary
from .model_monitor_summary import ModelMonitorSummary
from .model_monitor_summary_from_org import ModelMonitorSummaryFromOrg
from .model_monitor_version import ModelMonitorVersion
from .model_monitor_version_metric_data import ModelMonitorVersionMetricData
from .model_training_type_for_deployment import ModelTrainingTypeForDeployment
from .model_version import ModelVersion
from .modification_lock_info import ModificationLockInfo
from .module import Module
from .monitor_alert import MonitorAlert
from .monitor_alert_version import MonitorAlertVersion
from .natural_language_explanation import NaturalLanguageExplanation
from .notebook_completion import NotebookCompletion
from .organization_group import OrganizationGroup
from .organization_search_result import OrganizationSearchResult
from .pipeline import Pipeline
from .pipeline_step import PipelineStep
from .pipeline_step_version_logs import PipelineStepVersionLogs
from .pipeline_version import PipelineVersion
from .pipeline_version_logs import PipelineVersionLogs
from .problem_type import ProblemType
from .project import Project
from .project_config import ProjectConfig
from .project_validation import ProjectValidation
from .python_function import PythonFunction
from .python_plot_function import PythonPlotFunction
from .refresh_pipeline_run import RefreshPipelineRun
from .refresh_policy import RefreshPolicy
from .resolved_feature_group_template import ResolvedFeatureGroupTemplate
from .return_class import AbstractApiClass
from .schema import Schema
from .streaming_auth_token import StreamingAuthToken
from .streaming_connector import StreamingConnector
from .test_point_predictions import TestPointPredictions
from .training_config_options import TrainingConfigOptions
from .upload import Upload
from .upload_part import UploadPart
from .use_case import UseCase
from .use_case_requirements import UseCaseRequirements
from .user import User
from .webhook import Webhook


DEFAULT_SERVER = 'https://api.abacus.ai'


_request_context = threading.local()


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


_cached_endpoints = {}


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

    def __init__(self, message: str, http_status: int, exception: str = None, request_id: str = None):
        self.message = message
        self.http_status = http_status
        self.exception = exception or 'ApiException'
        self.request_id = request_id

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
    client_version = '0.73.1'

    def __init__(self, api_key: str = None, server: str = None, client_options: ClientOptions = None, skip_version_check: bool = False):
        self.api_key = api_key
        if self.api_key is None:
            self.api_key = os.getenv('ABACUS_API_KEY')
        self.notebook_id = os.getenv('ABACUS_NOTEBOOK_ID')
        self.web_version = None
        self.api_endpoint = None
        self.prediction_endpoint = None
        if not client_options:
            client_options = ClientOptions(server=os.getenv(
                'ABACUS_API_SERVER')) if os.getenv('ABACUS_API_SERVER') else ClientOptions()
        self.client_options = client_options
        self.server = server or self.client_options.server
        # Deprecated
        self.service_discovery_url = None
        # Connection and version check
        if api_key is not None:
            try:
                endpoint_info = self._call_api('getApiEndpoint', 'GET')
                self.prediction_endpoint = endpoint_info['predictEndpoint']
                if not self.server:
                    self.server = endpoint_info['apiEndpoint']
            except Exception:
                logging.error('Invalid API Key')
        skip_version_check = skip_version_check or self.client_version.startswith(
            'g')
        if not skip_version_check:
            try:
                self.web_version = self._call_api('version', 'GET')
                if version.parse(self.web_version) > version.parse(self.client_version):
                    warnings.warn(
                        'A new version of the Abacus.AI library is available')
                    warnings.warn(
                        f'Current Version: {self.client_version} -> New Version: {self.web_version}')
            except Exception:
                logging.error(
                    f'Failed get the current API version from Abacus.AI ({self.server or DEFAULT_SERVER}/api/v0/version)')

    def _get_prediction_endpoint(self, deployment_id: str, deployment_token: str):
        if self.prediction_endpoint:
            return self.prediction_endpoint
        if self.service_discovery_url:
            return None
        global _cached_endpoints
        cache_key = deployment_id + '|' + deployment_token
        endpoint_info = _cached_endpoints.get(cache_key)
        if not endpoint_info:
            endpoint_info = self._call_api('getApiEndpoint', 'GET', query_params={
                                           'deploymentId': deployment_id, 'deploymentToken': deployment_token})
            _cached_endpoints[cache_key] = endpoint_info
        return endpoint_info.get('predictEndpoint')

    def _get_streaming_endpoint(self, streaming_token: str, model_id: str = None, feature_group_id: str = None):
        if self.prediction_endpoint:
            return self.prediction_endpoint
        if self.service_discovery_url:
            return None
        global _cached_endpoints
        cache_key = (model_id or '') + (feature_group_id or '') + \
            '|' + streaming_token
        endpoint_info = _cached_endpoints.get(cache_key)
        if not endpoint_info:
            endpoint_info = self._call_api('getApiEndpoint', 'GET', query_params={
                                           'modelId': model_id, 'featureGroupId': feature_group_id, 'streamingToken': streaming_token})
            _cached_endpoints[cache_key] = endpoint_info
        return endpoint_info.get('predictEndpoint')

    def _clean_api_objects(self, obj):
        for key, val in (obj or {}).items():
            if isinstance(val, StreamingAuthToken):
                obj[key] = val.streaming_token
            elif isinstance(val, DeploymentAuthToken):
                obj[key] = val.deployment_token
            elif isinstance(val, AbstractApiClass):
                obj[key] = getattr(val, 'id', None)
            elif isinstance(val, ApiClass):
                obj[key] = val.to_dict()
            elif isinstance(val, list):
                obj[key] = [val_elem.to_dict() if isinstance(
                    val_elem, ApiClass) else val_elem for val_elem in val]
            elif isinstance(val, dict):
                obj[key] = {dict_key: dict_val.to_dict() if isinstance(
                    dict_val, ApiClass) else dict_val for dict_key, dict_val in val.items()}
            elif isinstance(val, ApiEnum):
                obj[key] = val.value
            elif callable(val):
                try:
                    obj[key] = inspect.getsource(val)
                except OSError:
                    raise OSError(
                        f'Could not get source for function {key}. Please pass a stringified version of this function when the function is defined in a shell environment.')

    def _call_api(
            self, action, method, query_params=None,
            body=None, files=None, parse_type=None, streamable_response=False, server_override=None, timeout=None):
        headers = {'apiKey': self.api_key, 'clientVersion': self.client_version,
                   'User-Agent': f'python-abacusai-{self.client_version}', 'notebookId': self.notebook_id}
        url = (server_override or self.server) + '/api/v0/' + action
        self._clean_api_objects(query_params)
        self._clean_api_objects(body)
        if self.service_discovery_url:
            discovered_url = _discover_service_url(self.service_discovery_url, self.client_version, query_params.get(
                'deploymentId') if query_params else None, query_params.get('deploymentToken') if query_params else None)
            if discovered_url:
                url = discovered_url + '/api/' + action
        response = self._request(url, method, query_params=query_params, headers=headers,
                                 body=body, files=files, stream=streamable_response, timeout=timeout)

        result = None
        success = False
        error_message = None
        error_type = None
        json_data = None
        request_id = None
        if streamable_response and response.status_code == 200:
            return response.raw
        try:
            json_data = response.json()
            success = json_data['success']
            error_message = json_data.get('error')
            error_type = json_data.get('errorType')
            request_id = json_data.get('requestId')
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
            elif response.status_code == 429:
                error_message = 'Too many requests. Please slow down your API requests'
            elif response.status_code > 502 and response.status_code not in (501, 503):
                error_message = 'Internal Server Error, please contact dev@abacus.ai for support'
            elif response.status_code == 404 and not self.client_options.exception_on_404:
                return None
            if request_id is None and response.headers:
                request_id = response.headers.get('x-request-id')
            if request_id:
                error_message += f". Request ID: {request_id}"
            raise ApiException(
                error_message, response.status_code, error_type, request_id)
        return result

    def _build_class(self, return_class, values):
        if values is None or values == {}:
            return None
        if isinstance(values, list):
            return [self._build_class(return_class, val) for val in values if val is not None]
        type_inputs = inspect.signature(return_class.__init__).parameters
        return return_class(self, **{key: val for key, val in values.items() if key in type_inputs})

    def _request(self, url, method, query_params=None, headers=None,
                 body=None, files=None, stream=False, timeout=None):
        if method == 'GET':
            cleaned_params = {key: ','.join([str(item) for item in val]) if isinstance(
                val, list) else val for key, val in query_params.items()} if query_params else query_params
            return _requests_retry_session().get(url, params=cleaned_params, headers=headers, stream=stream)
        elif method == 'POST':
            return _requests_retry_session().post(url, params=query_params, json=body, headers=headers, files=files, timeout=timeout or 200)
        elif method == 'PUT':
            return _requests_retry_session().put(url, params=query_params, data=body, headers=headers, files=files, timeout=timeout or 200)
        elif method == 'PATCH':
            return _requests_retry_session().patch(url, params=query_params, json=body, headers=headers, files=files, timeout=timeout or 200)
        elif method == 'DELETE':
            return _requests_retry_session().delete(url, params=query_params, data=body, headers=headers)
        else:
            raise ValueError(
                'HTTP method must be `GET`, `POST`, `PATCH`, `PUT` or `DELETE`'
            )

    def _poll(self, obj, wait_states: set, delay: int = 15, timeout: int = 300, poll_args: dict = {}, status_field=None):
        start_time = time.time()
        while obj.get_status(**poll_args) in wait_states:
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f'Maximum wait time of {timeout}s exceeded')
            time.sleep(delay)
        return obj.refresh()

    def _validate_pandas_df(self, df, clean_column_names: bool):
        if clean_column_names:
            new_col_mapping = {}
            cleaned_cols = {}
            for col in df.columns:
                cleaned_col = clean_column_name(col)
                new_col_mapping[col] = cleaned_col
                if cleaned_col in cleaned_cols:
                    raise ValueError(
                        f'The following columns have the same cleaned column name: "{col}" and "{cleaned_cols[col]}". Please rename these columns such that they are not the same name when cleaned. To see the cleaned version of a column name, refer to the function api_client_utils.clean_column_name in the abacusai package.')
                cleaned_cols[cleaned_col] = col
            df = df.rename(columns=new_col_mapping)
        bad_column_names = [col for col in df.columns if bool(re.search(
            INVALID_PANDAS_COLUMN_NAME_CHARACTERS, col)) or not col[0] in string.ascii_letters]
        if bad_column_names:
            raise ValueError(
                f'The dataframe\'s Column(s): {bad_column_names} contain illegal characters. Please rename the columns such that they only contain alphanumeric characters and underscores, and must start with an alpha character.')
        return df

    def _upload_from_pandas(self, upload, df, clean_column_names=False) -> Dataset:
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
        """Lists all of the user's API keys

        Returns:
            list[ApiKey]: List of API Keys for the current user's organization."""
        return self._call_api('listApiKeys', 'GET', query_params={}, parse_type=ApiKey)

    def list_organization_users(self) -> List[User]:
        """Retrieves a list of all users in the organization, including pending users who have been invited.

        Returns:
            list[User]: An array of all the users in the organization."""
        return self._call_api('listOrganizationUsers', 'GET', query_params={}, parse_type=User)

    def describe_user(self) -> User:
        """Retrieve the current user's information, such as their name, email address, and admin status.

        Returns:
            User: An object containing information about the current user."""
        return self._call_api('describeUser', 'GET', query_params={}, parse_type=User)

    def list_organization_groups(self) -> List[OrganizationGroup]:
        """Lists all Organizations Groups

        Returns:
            list[OrganizationGroup]: A list of all the organization groups within this organization."""
        return self._call_api('listOrganizationGroups', 'GET', query_params={}, parse_type=OrganizationGroup)

    def describe_organization_group(self, organization_group_id: str) -> OrganizationGroup:
        """Returns the specific organization group passed in by the user.

        Args:
            organization_group_id (str): The unique identifier of the organization group to be described.

        Returns:
            OrganizationGroup: Information about a specific organization group."""
        return self._call_api('describeOrganizationGroup', 'GET', query_params={'organizationGroupId': organization_group_id}, parse_type=OrganizationGroup)

    def describe_webhook(self, webhook_id: str) -> Webhook:
        """Describe the webhook with a given ID.

        Args:
            webhook_id (str): Unique string identifier of the target webhook.

        Returns:
            Webhook: The webhook with the given ID."""
        return self._call_api('describeWebhook', 'GET', query_params={'webhookId': webhook_id}, parse_type=Webhook)

    def list_deployment_webhooks(self, deployment_id: str) -> List[Webhook]:
        """List all the webhooks attached to a given deployment.

        Args:
            deployment_id (str): Unique identifier of the target deployment.

        Returns:
            list[Webhook]: List of the webhooks attached to the given deployment ID."""
        return self._call_api('listDeploymentWebhooks', 'GET', query_params={'deploymentId': deployment_id}, parse_type=Webhook)

    def list_use_cases(self) -> List[UseCase]:
        """Retrieves a list of all use cases with descriptions. Use the given mappings to specify a use case when needed.

        Returns:
            list[UseCase]: A list of `UseCase` objects describing all the use cases addressed by the platform. For details, please refer to."""
        return self._call_api('listUseCases', 'GET', query_params={}, parse_type=UseCase)

    def describe_problem_type(self, problem_type: str) -> ProblemType:
        """Describes a problem type

        Args:
            problem_type (str): The problem type to get details on

        Returns:
            ProblemType: The problem type requirements"""
        return self._call_api('describeProblemType', 'GET', query_params={'problemType': problem_type}, parse_type=ProblemType)

    def describe_use_case_requirements(self, use_case: str) -> List[UseCaseRequirements]:
        """This API call returns the feature requirements for a specified use case.

        Args:
            use_case (str): This contains the Enum String for the use case whose dataset requirements are needed.

        Returns:
            list[UseCaseRequirements]: The feature requirements of the use case are returned, including all the feature groups required for the use case along with their descriptions and feature mapping details."""
        return self._call_api('describeUseCaseRequirements', 'GET', query_params={'useCase': use_case}, parse_type=UseCaseRequirements)

    def describe_project(self, project_id: str) -> Project:
        """Returns a description of a project.

        Args:
            project_id (str): A unique string identifier for the project.

        Returns:
            Project: The description of the project."""
        return self._call_api('describeProject', 'GET', query_params={'projectId': project_id}, parse_type=Project)

    def list_projects(self, limit: int = 100, start_after_id: str = None) -> List[Project]:
        """Retrieves a list of all projects in the current organization.

        Args:
            limit (int): The maximum length of the list of projects.
            start_after_id (str): The ID of the project after which the list starts.

        Returns:
            list[Project]: A list of all projects in the Organization the user is currently logged in to."""
        return self._call_api('listProjects', 'GET', query_params={'limit': limit, 'startAfterId': start_after_id}, parse_type=Project)

    def get_project_feature_group_config(self, feature_group_id: str, project_id: str) -> ProjectConfig:
        """Gets a feature group's project config

        Args:
            feature_group_id (str): Unique string identifier for the feature group.
            project_id (str): Unique string identifier for the project.

        Returns:
            ProjectConfig: The feature group's project configuration."""
        return self._call_api('getProjectFeatureGroupConfig', 'GET', query_params={'featureGroupId': feature_group_id, 'projectId': project_id}, parse_type=ProjectConfig)

    def validate_project(self, project_id: str, feature_group_ids: list = None) -> ProjectValidation:
        """Validates that the specified project has all required feature group types for its use case and that all required feature columns are set.

        Args:
            project_id (str): The unique ID associated with the project.
            feature_group_ids (list): The list of feature group IDs to validate.

        Returns:
            ProjectValidation: The project validation. If the specified project is missing required columns or feature groups, the response includes an array of objects for each missing required feature group and the missing required features in each feature group."""
        return self._call_api('validateProject', 'GET', query_params={'projectId': project_id, 'featureGroupIds': feature_group_ids}, parse_type=ProjectValidation)

    def infer_feature_mappings(self, project_id: str, feature_group_id: str) -> InferredFeatureMappings:
        """Infer the feature mappings for the feature group in the project based on the problem type.

        Args:
            project_id (str): The unique ID associated with the project.
            feature_group_id (str): The unique ID associated with the feature group.

        Returns:
            InferredFeatureMappings: A dict that contains the inferred feature mappings."""
        return self._call_api('inferFeatureMappings', 'GET', query_params={'projectId': project_id, 'featureGroupId': feature_group_id}, parse_type=InferredFeatureMappings)

    def verify_and_describe_annotation(self, feature_group_id: str, feature_name: str = None, doc_id: str = None, feature_group_row_identifier: str = None) -> AnnotationEntry:
        """Get the latest annotation entry for a given feature group, feature, and document along with verification information.

        Args:
            feature_group_id (str): The ID of the feature group the annotation is on.
            feature_name (str): The name of the feature the annotation is on.
            doc_id (str): The ID of the primary document the annotation is on. At least one of the doc_id or feature_group_row_identifier must be provided in order to identify the correct annotation.
            feature_group_row_identifier (str): The key value of the feature group row the annotation is on (cast to string). Usually the feature group's primary / identifier key value. At least one of the doc_id or feature_group_row_identifier must be provided in order to identify the correct annotation.

        Returns:
            AnnotationEntry: The latest annotation entry for the given feature group, feature, document, and/or annotation key value. Includes the verification information."""
        return self._call_api('verifyAndDescribeAnnotation', 'GET', query_params={'featureGroupId': feature_group_id, 'featureName': feature_name, 'docId': doc_id, 'featureGroupRowIdentifier': feature_group_row_identifier}, parse_type=AnnotationEntry)

    def get_annotations_status(self, feature_group_id: str, feature_name: str = None, check_for_materialization: bool = False) -> AnnotationsStatus:
        """Get the status of the annotations for a given feature group and feature.

        Args:
            feature_group_id (str): The ID of the feature group the annotation is on.
            feature_name (str): The name of the feature the annotation is on.
            check_for_materialization (bool): If True, check if the feature group needs to be materialized before using for annotations.

        Returns:
            AnnotationsStatus: The status of the annotations for the given feature group and feature."""
        return self._call_api('getAnnotationsStatus', 'GET', query_params={'featureGroupId': feature_group_id, 'featureName': feature_name, 'checkForMaterialization': check_for_materialization}, parse_type=AnnotationsStatus)

    def get_feature_group_schema(self, feature_group_id: str, project_id: str = None) -> List[Feature]:
        """Returns a schema for a given FeatureGroup in a project.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            project_id (str): The unique ID associated with the project.

        Returns:
            list[Feature]: A list of objects for each column in the specified feature group."""
        return self._call_api('getFeatureGroupSchema', 'GET', query_params={'featureGroupId': feature_group_id, 'projectId': project_id}, parse_type=Feature)

    def get_point_in_time_feature_group_creation_options(self) -> List[GeneratedPitFeatureConfigOption]:
        """Returns the options that can be used to generate PIT features.

        Returns:
            list[GeneratedPitFeatureConfigOption]: List of possible generated aggregation function options."""
        return self._call_api('getPointInTimeFeatureGroupCreationOptions', 'GET', query_params={}, parse_type=GeneratedPitFeatureConfigOption)

    def describe_feature_group(self, feature_group_id: str) -> FeatureGroup:
        """Describe a Feature Group.

        Args:
            feature_group_id (str): A unique string identifier associated with the feature group.

        Returns:
            FeatureGroup: The feature group object."""
        return self._call_api('describeFeatureGroup', 'GET', query_params={'featureGroupId': feature_group_id}, parse_type=FeatureGroup)

    def describe_feature_group_by_table_name(self, table_name: str) -> FeatureGroup:
        """Describe a Feature Group by its table name.

        Args:
            table_name (str): The unique table name of the Feature Group to look up.

        Returns:
            FeatureGroup: The Feature Group."""
        return self._call_api('describeFeatureGroupByTableName', 'GET', query_params={'tableName': table_name}, parse_type=FeatureGroup)

    def list_feature_groups(self, limit: int = 100, start_after_id: str = None, feature_group_template_id: str = None, is_including_detached_from_template: bool = False) -> List[FeatureGroup]:
        """List all the feature groups

        Args:
            limit (int): The number of feature groups to retrieve.
            start_after_id (str): An offset parameter to exclude all feature groups up to a specified ID.
            feature_group_template_id (str): If specified, limit the results to feature groups attached to this template ID.
            is_including_detached_from_template (bool): When feature_group_template_id is specified, include feature groups that have been detached from that template ID.

        Returns:
            list[FeatureGroup]: All the feature groups in the organization associated with the specified project."""
        return self._call_api('listFeatureGroups', 'GET', query_params={'limit': limit, 'startAfterId': start_after_id, 'featureGroupTemplateId': feature_group_template_id, 'isIncludingDetachedFromTemplate': is_including_detached_from_template}, parse_type=FeatureGroup)

    def describe_project_feature_group(self, project_id: str, feature_group_id: str) -> FeatureGroup:
        """Describe a feature group associated with a project

        Args:
            project_id (str): The unique ID associated with the project.
            feature_group_id (str): The unique ID associated with the feature group.

        Returns:
            FeatureGroup: The feature group object."""
        return self._call_api('describeProjectFeatureGroup', 'GET', query_params={'projectId': project_id, 'featureGroupId': feature_group_id}, parse_type=FeatureGroup)

    def list_project_feature_groups(self, project_id: str, filter_feature_group_use: str = None) -> List[FeatureGroup]:
        """List all the feature groups associated with a project

        Args:
            project_id (str): The unique ID associated with the project.
            filter_feature_group_use (str): The feature group use filter, when given as an argument, only allows feature groups present in this project to be returned if they are of the given use. Possible values are:  DATA_WRANGLING,  TRAINING_INPUT,  BATCH_PREDICTION_INPUT,  BATCH_PREDICTION_OUTPUT.

        Returns:
            list[FeatureGroup]: All the Feature Groups in a project."""
        return self._call_api('listProjectFeatureGroups', 'GET', query_params={'projectId': project_id, 'filterFeatureGroupUse': filter_feature_group_use}, parse_type=FeatureGroup)

    def list_python_function_feature_groups(self, name: str, limit: int = 100) -> List[FeatureGroup]:
        """List all the feature groups associated with a python function.

        Args:
            name (str): The name used to identify the Python function.
            limit (int): The maximum number of feature groups to be retrieved.

        Returns:
            list[FeatureGroup]: All the feature groups associated with the specified Python function ID."""
        return self._call_api('listPythonFunctionFeatureGroups', 'GET', query_params={'name': name, 'limit': limit}, parse_type=FeatureGroup)

    def execute_async_feature_group_operation(self, query: str = None) -> ExecuteFeatureGroupOperation:
        """Starts the execution of fg operation

        Args:
            query (str): The SQL to be executed.

        Returns:
            ExecuteFeatureGroupOperation: A dict that contains the execution status"""
        return self._call_api('executeAsyncFeatureGroupOperation', 'GET', query_params={'query': query}, parse_type=ExecuteFeatureGroupOperation)

    def get_execute_feature_group_operation_result_part_count(self, feature_group_operation_run_id: str) -> int:
        """Gets the number of parts in the result of the execution of fg operation

        Args:
            feature_group_operation_run_id (str): The unique ID associated with the execution."""
        return self._call_api('getExecuteFeatureGroupOperationResultPartCount', 'GET', query_params={'featureGroupOperationRunId': feature_group_operation_run_id})

    def download_execute_feature_group_operation_result_part_chunk(self, feature_group_operation_run_id: str, part: int, offset: int = 0, chunk_size: int = 10485760) -> io.BytesIO:
        """Downloads a chunk of the result of the execution of fg operation

        Args:
            feature_group_operation_run_id (str): The unique ID associated with the execution.
            part (int): The part number of the result
            offset (int): The offset in the part
            chunk_size (int): The size of the chunk"""
        return self._call_api('downloadExecuteFeatureGroupOperationResultPartChunk', 'GET', query_params={'featureGroupOperationRunId': feature_group_operation_run_id, 'part': part, 'offset': offset, 'chunkSize': chunk_size}, streamable_response=True)

    def get_feature_group_version_export_download_url(self, feature_group_export_id: str) -> FeatureGroupExportDownloadUrl:
        """Get a link to download the feature group version.

        Args:
            feature_group_export_id (str): Unique identifier of the Feature Group Export to get a signed URL for.

        Returns:
            FeatureGroupExportDownloadUrl: Instance containing the download URL and expiration time for the Feature Group Export."""
        return self._call_api('getFeatureGroupVersionExportDownloadUrl', 'GET', query_params={'featureGroupExportId': feature_group_export_id}, parse_type=FeatureGroupExportDownloadUrl)

    def describe_feature_group_export(self, feature_group_export_id: str) -> FeatureGroupExport:
        """A feature group export

        Args:
            feature_group_export_id (str): Unique identifier of the feature group export.

        Returns:
            FeatureGroupExport: The feature group export object."""
        return self._call_api('describeFeatureGroupExport', 'GET', query_params={'featureGroupExportId': feature_group_export_id}, parse_type=FeatureGroupExport)

    def list_feature_group_exports(self, feature_group_id: str) -> List[FeatureGroupExport]:
        """Lists all of the feature group exports for the feature group

        Args:
            feature_group_id (str): Unique identifier of the feature group

        Returns:
            list[FeatureGroupExport]: List of feature group exports"""
        return self._call_api('listFeatureGroupExports', 'GET', query_params={'featureGroupId': feature_group_id}, parse_type=FeatureGroupExport)

    def get_feature_group_export_connector_errors(self, feature_group_export_id: str) -> io.BytesIO:
        """Returns a stream containing the write errors of the feature group export database connection, if any writes failed to the database connector.

        Args:
            feature_group_export_id (str): Unique identifier of the feature group export to get the errors for."""
        return self._call_api('getFeatureGroupExportConnectorErrors', 'GET', query_params={'featureGroupExportId': feature_group_export_id}, streamable_response=True)

    def list_feature_group_modifiers(self, feature_group_id: str) -> ModificationLockInfo:
        """List the users who can modify a given feature group.

        Args:
            feature_group_id (str): Unique string identifier of the feature group.

        Returns:
            ModificationLockInfo: Information about the modification lock status and groups/organizations added to the feature group."""
        return self._call_api('listFeatureGroupModifiers', 'GET', query_params={'featureGroupId': feature_group_id}, parse_type=ModificationLockInfo)

    def get_materialization_logs(self, feature_group_version: str, stdout: bool = False, stderr: bool = False) -> FunctionLogs:
        """Returns logs for a materialized feature group version.

        Args:
            feature_group_version (str): Unique string identifier for the feature group instance to export.
            stdout (bool): Set to True to get info logs.
            stderr (bool): Set to True to get error logs.

        Returns:
            FunctionLogs: A function logs object."""
        return self._call_api('getMaterializationLogs', 'GET', query_params={'featureGroupVersion': feature_group_version, 'stdout': stdout, 'stderr': stderr}, parse_type=FunctionLogs)

    def list_feature_group_versions(self, feature_group_id: str, limit: int = 100, start_after_version: str = None) -> List[FeatureGroupVersion]:
        """Retrieves a list of all feature group versions for the specified feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            limit (int): The maximum length of the returned versions.
            start_after_version (str): Results will start after this version.

        Returns:
            list[FeatureGroupVersion]: A list of feature group versions."""
        return self._call_api('listFeatureGroupVersions', 'GET', query_params={'featureGroupId': feature_group_id, 'limit': limit, 'startAfterVersion': start_after_version}, parse_type=FeatureGroupVersion)

    def describe_feature_group_version(self, feature_group_version: str) -> FeatureGroupVersion:
        """Describe a feature group version.

        Args:
            feature_group_version (str): The unique identifier associated with the feature group version.

        Returns:
            FeatureGroupVersion: The feature group version."""
        return self._call_api('describeFeatureGroupVersion', 'GET', query_params={'featureGroupVersion': feature_group_version}, parse_type=FeatureGroupVersion)

    def get_feature_group_version_metrics(self, feature_group_version: str, selected_columns: list = None, include_charts: bool = False, include_statistics: bool = True) -> DataMetrics:
        """Get metrics for a specific feature group version.

        Args:
            feature_group_version (str): A unique string identifier associated with the feature group version.
            selected_columns (list): A list of columns to order first.
            include_charts (bool): A flag indicating whether charts should be included in the response. Default is false.
            include_statistics (bool): A flag indicating whether statistics should be included in the response. Default is true.

        Returns:
            DataMetrics: The metrics for the specified feature group version."""
        return self._call_api('getFeatureGroupVersionMetrics', 'GET', query_params={'featureGroupVersion': feature_group_version, 'selectedColumns': selected_columns, 'includeCharts': include_charts, 'includeStatistics': include_statistics}, parse_type=DataMetrics)

    def describe_feature_group_template(self, feature_group_template_id: str) -> FeatureGroupTemplate:
        """Describe a Feature Group Template.

        Args:
            feature_group_template_id (str): The unique identifier of a feature group template.

        Returns:
            FeatureGroupTemplate: The feature group template object."""
        return self._call_api('describeFeatureGroupTemplate', 'GET', query_params={'featureGroupTemplateId': feature_group_template_id}, parse_type=FeatureGroupTemplate)

    def list_feature_group_templates(self, limit: int = 100, start_after_id: str = None, feature_group_id: str = None, should_include_system_templates: bool = False) -> List[FeatureGroupTemplate]:
        """List feature group templates, optionally scoped by the feature group that created the templates.

        Args:
            limit (int): Maximum number of templates to be retrieved.
            start_after_id (str): Offset parameter to exclude all templates up to the specified feature group template ID.
            feature_group_id (str): If specified, limit to templates created from this feature group.
            should_include_system_templates (bool): If True, will include built-in templates.

        Returns:
            list[FeatureGroupTemplate]: All the feature groups in the organization, optionally limited by the feature group that created the template(s)."""
        return self._call_api('listFeatureGroupTemplates', 'GET', query_params={'limit': limit, 'startAfterId': start_after_id, 'featureGroupId': feature_group_id, 'shouldIncludeSystemTemplates': should_include_system_templates}, parse_type=FeatureGroupTemplate)

    def list_project_feature_group_templates(self, project_id: str, limit: int = 100, start_after_id: str = None, should_include_all_system_templates: bool = False) -> List[FeatureGroupTemplate]:
        """List feature group templates for feature groups associated with the project.

        Args:
            project_id (str): Unique string identifier to limit to templates associated with this project, e.g. templates associated with feature groups in this project.
            limit (int): Maximum number of templates to be retrieved.
            start_after_id (str): Offset parameter to exclude all templates till the specified feature group template ID.
            should_include_all_system_templates (bool): If True, will include built-in templates.

        Returns:
            list[FeatureGroupTemplate]: All the feature groups in the organization, optionally limited by the feature group that created the template(s)."""
        return self._call_api('listProjectFeatureGroupTemplates', 'GET', query_params={'projectId': project_id, 'limit': limit, 'startAfterId': start_after_id, 'shouldIncludeAllSystemTemplates': should_include_all_system_templates}, parse_type=FeatureGroupTemplate)

    def suggest_feature_group_template_for_feature_group(self, feature_group_id: str) -> FeatureGroupTemplate:
        """Suggest values for a feature gruop template, based on a feature group.

        Args:
            feature_group_id (str): Unique identifier associated with the feature group to use for suggesting values to use in the template.

        Returns:
            FeatureGroupTemplate: The suggested feature group template."""
        return self._call_api('suggestFeatureGroupTemplateForFeatureGroup', 'GET', query_params={'featureGroupId': feature_group_id}, parse_type=FeatureGroupTemplate)

    def get_dataset_schema(self, dataset_id: str) -> List[DatasetColumn]:
        """Retrieves the column schema of a dataset.

        Args:
            dataset_id (str): Unique string identifier of the dataset schema to look up.

        Returns:
            list[DatasetColumn]: List of column schema definitions."""
        return self._call_api('getDatasetSchema', 'GET', query_params={'datasetId': dataset_id}, parse_type=DatasetColumn)

    def set_dataset_database_connector_config(self, dataset_id: str, database_connector_id: str, object_name: str = None, columns: str = None, query_arguments: str = None, sql_query: str = None):
        """Sets database connector config for a dataset. This method is currently only supported for streaming datasets.

        Args:
            dataset_id (str): Unique String Identifier of the dataset_id.
            database_connector_id (str): Unique String Identifier of the Database Connector to import the dataset from.
            object_name (str): If applicable, the name/ID of the object in the service to query.
            columns (str): The columns to query from the external service object.
            query_arguments (str): Additional query arguments to filter the data.
            sql_query (str): The full SQL query to use when fetching data. If present, this parameter will override `object_name`, `columns` and `query_arguments`."""
        return self._call_api('setDatasetDatabaseConnectorConfig', 'GET', query_params={'datasetId': dataset_id, 'databaseConnectorId': database_connector_id, 'objectName': object_name, 'columns': columns, 'queryArguments': query_arguments, 'sqlQuery': sql_query})

    def get_dataset_version_metrics(self, dataset_version: str, selected_columns: list = None, include_charts: bool = False, include_statistics: bool = True) -> DataMetrics:
        """Get metrics for a specific dataset version.

        Args:
            dataset_version (str): A unique string identifier associated with the dataset version.
            selected_columns (list): A list of columns to order first.
            include_charts (bool): A flag indicating whether charts should be included in the response. Default is false.
            include_statistics (bool): A flag indicating whether statistics should be included in the response. Default is true.

        Returns:
            DataMetrics: The metrics for the specified Dataset version."""
        return self._call_api('getDatasetVersionMetrics', 'GET', query_params={'datasetVersion': dataset_version, 'selectedColumns': selected_columns, 'includeCharts': include_charts, 'includeStatistics': include_statistics}, parse_type=DataMetrics)

    def get_file_connector_instructions(self, bucket: str, write_permission: bool = False) -> FileConnectorInstructions:
        """Retrieves verification information to create a data connector to a cloud storage bucket.

        Args:
            bucket (str): The fully-qualified URI of the storage bucket to verify.
            write_permission (bool): If `True`, instructions will include steps for allowing Abacus.AI to write to this service.

        Returns:
            FileConnectorInstructions: An object with a full description of the cloud storage bucket authentication options and bucket policy. Returns an error message if the parameters are invalid."""
        return self._call_api('getFileConnectorInstructions', 'GET', query_params={'bucket': bucket, 'writePermission': write_permission}, parse_type=FileConnectorInstructions)

    def list_database_connectors(self) -> List[DatabaseConnector]:
        """Retrieves a list of all database connectors along with their associated attributes.

        Returns:
            list[DatabaseConnector]: An object containing the database connector and its attributes."""
        return self._call_api('listDatabaseConnectors', 'GET', query_params={}, parse_type=DatabaseConnector)

    def list_file_connectors(self) -> List[FileConnector]:
        """Retrieves a list of all connected services in the organization and their current verification status.

        Returns:
            list[FileConnector]: A list of cloud storage buckets connected to the organization."""
        return self._call_api('listFileConnectors', 'GET', query_params={}, parse_type=FileConnector)

    def list_database_connector_objects(self, database_connector_id: str) -> list:
        """Lists querable objects in the database connector.

        Args:
            database_connector_id (str): Unique string identifier for the database connector."""
        return self._call_api('listDatabaseConnectorObjects', 'GET', query_params={'databaseConnectorId': database_connector_id})

    def get_database_connector_object_schema(self, database_connector_id: str, object_name: str = None) -> list:
        """Get the schema of an object in an database connector.

        Args:
            database_connector_id (str): Unique string identifier for the database connector.
            object_name (str): Unique identifier for the object in the external system."""
        return self._call_api('getDatabaseConnectorObjectSchema', 'GET', query_params={'databaseConnectorId': database_connector_id, 'objectName': object_name})

    def list_application_connectors(self) -> List[ApplicationConnector]:
        """Retrieves a list of all application connectors along with their associated attributes.

        Returns:
            list[ApplicationConnector]: A list of application connectors."""
        return self._call_api('listApplicationConnectors', 'GET', query_params={}, parse_type=ApplicationConnector)

    def list_application_connector_objects(self, application_connector_id: str) -> list:
        """Lists querable objects in the application connector.

        Args:
            application_connector_id (str): Unique string identifier for the application connector."""
        return self._call_api('listApplicationConnectorObjects', 'GET', query_params={'applicationConnectorId': application_connector_id})

    def list_streaming_connectors(self) -> List[StreamingConnector]:
        """Retrieves a list of all streaming connectors along with their corresponding attributes.

        Returns:
            list[StreamingConnector]: A list of StreamingConnector objects."""
        return self._call_api('listStreamingConnectors', 'GET', query_params={}, parse_type=StreamingConnector)

    def list_streaming_tokens(self) -> List[StreamingAuthToken]:
        """Retrieves a list of all streaming tokens.

        Returns:
            list[StreamingAuthToken]: A list of streaming tokens and their associated attributes."""
        return self._call_api('listStreamingTokens', 'GET', query_params={}, parse_type=StreamingAuthToken)

    def get_recent_feature_group_streamed_data(self, feature_group_id: str):
        """Returns recently streamed data to a streaming feature group.

        Args:
            feature_group_id (str): Unique string identifier associated with the feature group."""
        return self._call_api('getRecentFeatureGroupStreamedData', 'GET', query_params={'featureGroupId': feature_group_id})

    def list_uploads(self) -> List[Upload]:
        """Lists all pending uploads

        Returns:
            list[Upload]: A list of ongoing uploads in the organization."""
        return self._call_api('listUploads', 'GET', query_params={}, parse_type=Upload)

    def describe_upload(self, upload_id: str) -> Upload:
        """Retrieves the current upload status (complete or inspecting) and the list of file parts uploaded for a specified dataset upload.

        Args:
            upload_id (str): The unique ID associated with the file uploaded or being uploaded in parts.

        Returns:
            Upload: Details associated with the large dataset file uploaded in parts."""
        return self._call_api('describeUpload', 'GET', query_params={'uploadId': upload_id}, parse_type=Upload)

    def list_datasets(self, limit: int = 100, start_after_id: str = None, exclude_streaming: bool = False) -> List[Dataset]:
        """Retrieves a list of all datasets in the organization.

        Args:
            limit (int): Maximum length of the list of datasets.
            start_after_id (str): ID of the dataset after which the list starts.
            exclude_streaming (bool): Exclude streaming datasets from the result.

        Returns:
            list[Dataset]: List of datasets."""
        return self._call_api('listDatasets', 'GET', query_params={'limit': limit, 'startAfterId': start_after_id, 'excludeStreaming': exclude_streaming}, parse_type=Dataset)

    def describe_dataset(self, dataset_id: str) -> Dataset:
        """Retrieves a full description of the specified dataset, with attributes such as its ID, name, source type, etc.

        Args:
            dataset_id (str): The unique ID associated with the dataset.

        Returns:
            Dataset: The dataset."""
        return self._call_api('describeDataset', 'GET', query_params={'datasetId': dataset_id}, parse_type=Dataset)

    def describe_dataset_version(self, dataset_version: str) -> DatasetVersion:
        """Retrieves a full description of the specified dataset version, including its ID, name, source type, and other attributes.

        Args:
            dataset_version (str): Unique string identifier associated with the dataset version.

        Returns:
            DatasetVersion: The dataset version."""
        return self._call_api('describeDatasetVersion', 'GET', query_params={'datasetVersion': dataset_version}, parse_type=DatasetVersion)

    def list_dataset_versions(self, dataset_id: str, limit: int = 100, start_after_version: str = None) -> List[DatasetVersion]:
        """Retrieves a list of all dataset versions for the specified dataset.

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            limit (int): The maximum length of the list of all dataset versions.
            start_after_version (str): The ID of the version after which the list starts.

        Returns:
            list[DatasetVersion]: A list of dataset versions."""
        return self._call_api('listDatasetVersions', 'GET', query_params={'datasetId': dataset_id, 'limit': limit, 'startAfterVersion': start_after_version}, parse_type=DatasetVersion)

    def get_docstore_document(self, doc_id: str) -> io.BytesIO:
        """Return a document store document by id.

        Args:
            doc_id (str): Unique Docstore string identifier for the document."""
        return self._call_api('getDocstoreDocument', 'GET', query_params={'docId': doc_id}, streamable_response=True)

    def get_docstore_image(self, doc_id: str, max_width: int = None, max_height: int = None) -> io.BytesIO:
        """Return a document store image by id.

        Args:
            doc_id (str): A unique Docstore string identifier for the image.
            max_width (int): Rescales the returned image so the width is less than or equal to the given maximum width, while preserving the aspect ratio.
            max_height (int): Rescales the returned image so the height is less than or equal to the given maximum height, while preserving the aspect ratio."""
        return self._call_api('getDocstoreImage', 'GET', query_params={'docId': doc_id, 'maxWidth': max_width, 'maxHeight': max_height}, streamable_response=True)

    def describe_train_test_data_split_feature_group(self, model_id: str) -> FeatureGroup:
        """Get the train and test data split for a trained model by its unique identifier. This is only supported for models with custom algorithms.

        Args:
            model_id (str): The unique ID of the model. By default, the latest model version will be returned if no version is specified.

        Returns:
            FeatureGroup: The feature group containing the training data and fold information."""
        return self._call_api('describeTrainTestDataSplitFeatureGroup', 'GET', query_params={'modelId': model_id}, parse_type=FeatureGroup)

    def describe_train_test_data_split_feature_group_version(self, model_version: str) -> FeatureGroupVersion:
        """Get the train and test data split for a trained model by model version. This is only supported for models with custom algorithms.

        Args:
            model_version (str): The unique version ID of the model version.

        Returns:
            FeatureGroupVersion: The feature group version containing the training data and folds information."""
        return self._call_api('describeTrainTestDataSplitFeatureGroupVersion', 'GET', query_params={'modelVersion': model_version}, parse_type=FeatureGroupVersion)

    def list_models(self, project_id: str) -> List[Model]:
        """Retrieves the list of models in the specified project.

        Args:
            project_id (str): Unique string identifier associated with the project.

        Returns:
            list[Model]: A list of models."""
        return self._call_api('listModels', 'GET', query_params={'projectId': project_id}, parse_type=Model)

    def describe_model(self, model_id: str) -> Model:
        """Retrieves a full description of the specified model.

        Args:
            model_id (str): Unique string identifier associated with the model.

        Returns:
            Model: Description of the model."""
        return self._call_api('describeModel', 'GET', query_params={'modelId': model_id}, parse_type=Model)

    def set_model_objective(self, model_version: str, metric: str):
        """Sets the best model for all model instances of the model based on the specified metric, and updates the training configuration to use the specified metric for any future model versions.

        Args:
            model_version (str): The model version to set as the best model.
            metric (str): The metric to use to determine the best model."""
        return self._call_api('setModelObjective', 'GET', query_params={'modelVersion': model_version, 'metric': metric})

    def get_model_metrics(self, model_id: str, model_version: str = None, baseline_metrics: bool = False, return_graphs: bool = False) -> ModelMetrics:
        """Retrieves a full list of metrics for the specified model.

        If only the model's unique identifier (model_id) is specified, the latest trained version of the model (model_version) is used.


        Args:
            model_id (str): Unique string identifier for the model.
            model_version (str): Version of the model.
            baseline_metrics (bool): If true, will also return the baseline model metrics for comparison.
            return_graphs (bool): If true, will return the information used for the graphs on the model metrics page.

        Returns:
            ModelMetrics: An object containing the model metrics and explanations for what each metric means."""
        return self._call_api('getModelMetrics', 'GET', query_params={'modelId': model_id, 'modelVersion': model_version, 'baselineMetrics': baseline_metrics, 'returnGraphs': return_graphs}, parse_type=ModelMetrics)

    def query_test_point_predictions(self, model_version: str, model_name: str, to_row: int, from_row: int = 0, sql_where_clause: str = '') -> TestPointPredictions:
        """Query the test points predictions data for a specific algorithm.

        Args:
            model_version (str): The unique ID associated with the model version.
            model_name (str): The model name
            to_row (int): Ending row index to return.
            from_row (int): Starting row index to return.
            sql_where_clause (str): The SQL WHERE clause used to filter the data.

        Returns:
            TestPointPredictions: TestPointPrediction"""
        return self._call_api('queryTestPointPredictions', 'GET', query_params={'modelVersion': model_version, 'modelName': model_name, 'toRow': to_row, 'fromRow': from_row, 'sqlWhereClause': sql_where_clause}, parse_type=TestPointPredictions)

    def list_model_versions(self, model_id: str, limit: int = 100, start_after_version: str = None) -> List[ModelVersion]:
        """Retrieves a list of versions for a given model.

        Args:
            model_id (str): Unique string identifier associated with the model.
            limit (int): Maximum length of the list of all dataset versions.
            start_after_version (str): Unique string identifier of the version after which the list starts.

        Returns:
            list[ModelVersion]: An array of model versions."""
        return self._call_api('listModelVersions', 'GET', query_params={'modelId': model_id, 'limit': limit, 'startAfterVersion': start_after_version}, parse_type=ModelVersion)

    def describe_model_version(self, model_version: str) -> ModelVersion:
        """Retrieves a full description of the specified model version.

        Args:
            model_version (str): Unique string identifier of the model version.

        Returns:
            ModelVersion: A model version."""
        return self._call_api('describeModelVersion', 'GET', query_params={'modelVersion': model_version}, parse_type=ModelVersion)

    def get_feature_importance_by_model_version(self, model_version: str) -> FeatureImportance:
        """Gets the feature importance calculated by various methods for the model.

        Args:
            model_version (str): Unique string identifier for the model version.

        Returns:
            FeatureImportance: Feature importances for the model."""
        return self._call_api('getFeatureImportanceByModelVersion', 'GET', query_params={'modelVersion': model_version}, parse_type=FeatureImportance)

    def get_training_data_logs(self, model_version: str) -> List[DataPrepLogs]:
        """Retrieves the data preparation logs during model training.

        Args:
            model_version (str): The unique version ID of the model version.

        Returns:
            list[DataPrepLogs]: A list of logs."""
        return self._call_api('getTrainingDataLogs', 'GET', query_params={'modelVersion': model_version}, parse_type=DataPrepLogs)

    def set_default_model_algorithm(self, model_id: str = None, algorithm: str = None, data_cluster_type: str = None):
        """Sets the model's algorithm to default for all new deployments

        Args:
            model_id (str): Unique identifier of the model to set.
            algorithm (str): Algorithm to pin in the model.
            data_cluster_type (str): Data cluster type to set the lead model for."""
        return self._call_api('setDefaultModelAlgorithm', 'GET', query_params={'modelId': model_id, 'algorithm': algorithm, 'dataClusterType': data_cluster_type})

    def get_training_logs(self, model_version: str, stdout: bool = False, stderr: bool = False) -> FunctionLogs:
        """Returns training logs for the model.

        Args:
            model_version (str): The unique version ID of the model version.
            stdout (bool): Set True to get info logs.
            stderr (bool): Set True to get error logs.

        Returns:
            FunctionLogs: A function logs object."""
        return self._call_api('getTrainingLogs', 'GET', query_params={'modelVersion': model_version, 'stdout': stdout, 'stderr': stderr}, parse_type=FunctionLogs)

    def export_custom_model_version(self, model_version: str, output_location: str, algorithm: str = None) -> ModelArtifactsExport:
        """Bundle custom model artifacts to a zip file, and export to the specified location.

        Args:
            model_version (str): A unique string identifier for the model version.
            output_location (str): Location to export the model artifacts results. For example, s3://a-bucket/
            algorithm (str): The algorithm to be exported. Optional if there's only one custom algorithm in the model version.

        Returns:
            ModelArtifactsExport: Object describing the export and its status."""
        return self._call_api('exportCustomModelVersion', 'GET', query_params={'modelVersion': model_version, 'outputLocation': output_location, 'algorithm': algorithm}, parse_type=ModelArtifactsExport)

    def describe_model_artifacts_export(self, model_artifacts_export_id: str) -> ModelArtifactsExport:
        """Get the description and status of the model artifacts export.

        Args:
            model_artifacts_export_id (str): A unique string identifier for the export.

        Returns:
            ModelArtifactsExport: Object describing the export and its status."""
        return self._call_api('describeModelArtifactsExport', 'GET', query_params={'modelArtifactsExportId': model_artifacts_export_id}, parse_type=ModelArtifactsExport)

    def list_model_artifacts_exports(self, model_id: str, limit: int = 25) -> List[ModelArtifactsExport]:
        """List all the model artifacts exports.

        Args:
            model_id (str): A unique string identifier for the model.
            limit (int): Maximum length of the list of all exports.

        Returns:
            list[ModelArtifactsExport]: List of model artifacts exports."""
        return self._call_api('listModelArtifactsExports', 'GET', query_params={'modelId': model_id, 'limit': limit}, parse_type=ModelArtifactsExport)

    def list_model_monitors(self, project_id: str) -> List[ModelMonitor]:
        """Retrieves the list of model monitors in the specified project.

        Args:
            project_id (str): Unique string identifier associated with the project.

        Returns:
            list[ModelMonitor]: A list of model monitors."""
        return self._call_api('listModelMonitors', 'GET', query_params={'projectId': project_id}, parse_type=ModelMonitor)

    def describe_model_monitor(self, model_monitor_id: str) -> ModelMonitor:
        """Retrieves a full description of the specified model monitor.

        Args:
            model_monitor_id (str): Unique string identifier associated with the model monitor.

        Returns:
            ModelMonitor: Description of the model monitor."""
        return self._call_api('describeModelMonitor', 'GET', query_params={'modelMonitorId': model_monitor_id}, parse_type=ModelMonitor)

    def get_prediction_drift(self, model_monitor_version: str) -> DriftDistributions:
        """Gets the label and prediction drifts for a model monitor.

        Args:
            model_monitor_version (str): Unique string identifier for a model monitor version created under the project.

        Returns:
            DriftDistributions: Object describing training and prediction output label and prediction distributions."""
        return self._call_api('getPredictionDrift', 'GET', query_params={'modelMonitorVersion': model_monitor_version}, parse_type=DriftDistributions)

    def get_model_monitor_summary(self, model_monitor_id: str) -> ModelMonitorSummary:
        """Gets the summary of a model monitor across versions.

        Args:
            model_monitor_id (str): A unique string identifier associated with the model monitor.

        Returns:
            ModelMonitorSummary: An object describing integrity, bias violations, model accuracy and drift for the model monitor."""
        return self._call_api('getModelMonitorSummary', 'GET', query_params={'modelMonitorId': model_monitor_id}, parse_type=ModelMonitorSummary)

    def list_model_monitor_versions(self, model_monitor_id: str, limit: int = 100, start_after_version: str = None) -> List[ModelMonitorVersion]:
        """Retrieves a list of versions for a given model monitor.

        Args:
            model_monitor_id (str): The unique ID associated with the model monitor.
            limit (int): The maximum length of the list of all model monitor versions.
            start_after_version (str): The ID of the version after which the list starts.

        Returns:
            list[ModelMonitorVersion]: A list of model monitor versions."""
        return self._call_api('listModelMonitorVersions', 'GET', query_params={'modelMonitorId': model_monitor_id, 'limit': limit, 'startAfterVersion': start_after_version}, parse_type=ModelMonitorVersion)

    def describe_model_monitor_version(self, model_monitor_version: str) -> ModelMonitorVersion:
        """Retrieves a full description of the specified model monitor version.

        Args:
            model_monitor_version (str): The unique version ID of the model monitor version.

        Returns:
            ModelMonitorVersion: A model monitor version."""
        return self._call_api('describeModelMonitorVersion', 'GET', query_params={'modelMonitorVersion': model_monitor_version}, parse_type=ModelMonitorVersion)

    def model_monitor_version_metric_data(self, model_monitor_version: str, metric_type: str, actual_values_to_detail: list = None) -> ModelMonitorVersionMetricData:
        """Provides the data needed for decile metrics associated with the model monitor.

        Args:
            model_monitor_version (str): Unique string identifier for the model monitor version.
            metric_type (str): The type of metric to get data for.
            actual_values_to_detail (list): The actual values to detail.

        Returns:
            ModelMonitorVersionMetricData: Data associated with the metric."""
        return self._call_api('modelMonitorVersionMetricData', 'GET', query_params={'modelMonitorVersion': model_monitor_version, 'metricType': metric_type, 'actualValuesToDetail': actual_values_to_detail}, parse_type=ModelMonitorVersionMetricData)

    def list_organization_model_monitors(self, only_starred: bool = False) -> List[ModelMonitor]:
        """Gets a list of Model Monitors for an organization.

        Args:
            only_starred (bool): Whether to return only starred Model Monitors. Defaults to False.

        Returns:
            list[ModelMonitor]: A list of Model Monitors."""
        return self._call_api('listOrganizationModelMonitors', 'GET', query_params={'onlyStarred': only_starred}, parse_type=ModelMonitor)

    def get_model_monitor_chart_from_organization(self, chart_type: str, limit: int = 15) -> List[ModelMonitorSummaryFromOrg]:
        """Gets a list of model monitor summaries across monitors for an organization.

        Args:
            chart_type (str): Type of chart (model_accuracy, bias_violations, data_integrity, or model_drift) to return.
            limit (int): Maximum length of the model monitors.

        Returns:
            list[ModelMonitorSummaryFromOrg]: List of ModelMonitorSummaryForOrganization objects describing accuracy, bias, drift, or integrity for all model monitors in an organization."""
        return self._call_api('getModelMonitorChartFromOrganization', 'GET', query_params={'chartType': chart_type, 'limit': limit}, parse_type=ModelMonitorSummaryFromOrg)

    def get_model_monitor_summary_from_organization(self) -> List[ModelMonitorOrgSummary]:
        """Gets a consolidated summary of model monitors for an organization.

        Returns:
            list[ModelMonitorOrgSummary]: A list of `ModelMonitorSummaryForOrganization` objects describing accuracy, bias, drift, and integrity for all model monitors in an organization."""
        return self._call_api('getModelMonitorSummaryFromOrganization', 'GET', query_params={}, parse_type=ModelMonitorOrgSummary)

    def list_eda(self, project_id: str) -> List[Eda]:
        """Retrieves the list of Exploratory Data Analysis (EDA) in the specified project.

        Args:
            project_id (str): Unique string identifier associated with the project.

        Returns:
            list[Eda]: List of EDA objects."""
        return self._call_api('listEda', 'GET', query_params={'projectId': project_id}, parse_type=Eda)

    def describe_eda(self, eda_id: str) -> Eda:
        """Retrieves a full description of the specified EDA object.

        Args:
            eda_id (str): Unique string identifier associated with the EDA object.

        Returns:
            Eda: Description of the EDA object."""
        return self._call_api('describeEda', 'GET', query_params={'edaId': eda_id}, parse_type=Eda)

    def list_eda_versions(self, eda_id: str, limit: int = 100, start_after_version: str = None) -> List[EdaVersion]:
        """Retrieves a list of versions for a given EDA object.

        Args:
            eda_id (str): The unique ID associated with the EDA object.
            limit (int): The maximum length of the list of all EDA versions.
            start_after_version (str): The ID of the version after which the list starts.

        Returns:
            list[EdaVersion]: A list of EDA versions."""
        return self._call_api('listEdaVersions', 'GET', query_params={'edaId': eda_id, 'limit': limit, 'startAfterVersion': start_after_version}, parse_type=EdaVersion)

    def describe_eda_version(self, eda_version: str) -> EdaVersion:
        """Retrieves a full description of the specified EDA version.

        Args:
            eda_version (str): Unique string identifier of the EDA version.

        Returns:
            EdaVersion: An EDA version."""
        return self._call_api('describeEdaVersion', 'GET', query_params={'edaVersion': eda_version}, parse_type=EdaVersion)

    def get_eda_collinearity(self, eda_version: str) -> EdaCollinearity:
        """Gets the Collinearity between all features for the Exploratory Data Analysis.

        Args:
            eda_version (str): Unique string identifier associated with the EDA instance.

        Returns:
            EdaCollinearity: An object with a record of correlations between each feature for the EDA."""
        return self._call_api('getEdaCollinearity', 'GET', query_params={'edaVersion': eda_version}, parse_type=EdaCollinearity)

    def get_eda_data_consistency(self, eda_version: str, transformation_feature: str = None) -> EdaDataConsistency:
        """Gets the data consistency for the Exploratory Data Analysis.

        Args:
            eda_version (str): Unique string identifier associated with the EDA instance.
            transformation_feature (str): The transformation feature to get consistency for.

        Returns:
            EdaDataConsistency: Object with duplication, deletion, and transformation data for data consistency analysis for an EDA."""
        return self._call_api('getEdaDataConsistency', 'GET', query_params={'edaVersion': eda_version, 'transformationFeature': transformation_feature}, parse_type=EdaDataConsistency)

    def get_collinearity_for_feature(self, eda_version: str, feature_name: str = None) -> EdaFeatureCollinearity:
        """Gets the Collinearity for the given feature from the Exploratory Data Analysis.

        Args:
            eda_version (str): Unique string identifier associated with the EDA instance.
            feature_name (str): Name of the feature for which correlation is shown.

        Returns:
            EdaFeatureCollinearity: Object with a record of correlations for the provided feature for an EDA."""
        return self._call_api('getCollinearityForFeature', 'GET', query_params={'edaVersion': eda_version, 'featureName': feature_name}, parse_type=EdaFeatureCollinearity)

    def get_feature_association(self, eda_version: str, reference_feature_name: str, test_feature_name: str) -> EdaFeatureAssociation:
        """Gets the Feature Association for the given features from the feature group version within the eda_version.

        Args:
            eda_version (str): Unique string identifier associated with the EDA instance.
            reference_feature_name (str): Name of the feature for feature association (on x-axis for the plots generated for the Feature association in the product).
            test_feature_name (str): Name of the feature for feature association (on y-axis for the plots generated for the Feature association in the product).

        Returns:
            EdaFeatureAssociation: An object with a record of data for the feature association between the two given features for an EDA version."""
        return self._call_api('getFeatureAssociation', 'GET', query_params={'edaVersion': eda_version, 'referenceFeatureName': reference_feature_name, 'testFeatureName': test_feature_name}, parse_type=EdaFeatureAssociation)

    def get_eda_forecasting_analysis(self, eda_version: str) -> EdaForecastingAnalysis:
        """Gets the Forecasting analysis for the Exploratory Data Analysis.

        Args:
            eda_version (str): Unique string identifier associated with the EDA version.

        Returns:
            EdaForecastingAnalysis: Object with forecasting analysis that includes sales_across_time, cummulative_contribution, missing_value_distribution, history_length, num_rows_histogram, product_maturity data."""
        return self._call_api('getEdaForecastingAnalysis', 'GET', query_params={'edaVersion': eda_version}, parse_type=EdaForecastingAnalysis)

    def describe_monitor_alert(self, monitor_alert_id: str) -> MonitorAlert:
        """Describes a given monitor alert id

        Args:
            monitor_alert_id (str): Unique identifier of the monitor alert.

        Returns:
            MonitorAlert: Object containing information about the monitor alert."""
        return self._call_api('describeMonitorAlert', 'GET', query_params={'monitorAlertId': monitor_alert_id}, parse_type=MonitorAlert)

    def describe_monitor_alert_version(self, monitor_alert_version: str) -> MonitorAlertVersion:
        """Describes a given monitor alert version id

        Args:
            monitor_alert_version (str): Unique string identifier for the monitor alert.

        Returns:
            MonitorAlertVersion: An object describing the monitor alert version."""
        return self._call_api('describeMonitorAlertVersion', 'GET', query_params={'monitorAlertVersion': monitor_alert_version}, parse_type=MonitorAlertVersion)

    def list_monitor_alerts_for_monitor(self, model_monitor_id: str) -> List[MonitorAlert]:
        """Retrieves the list of monitor alerts for a specified monitor.

        Args:
            model_monitor_id (str): The unique ID associated with the model monitor.

        Returns:
            list[MonitorAlert]: A list of monitor alerts."""
        return self._call_api('listMonitorAlertsForMonitor', 'GET', query_params={'modelMonitorId': model_monitor_id}, parse_type=MonitorAlert)

    def list_monitor_alert_versions_for_monitor_version(self, model_monitor_version: str) -> List[MonitorAlertVersion]:
        """Retrieves the list of monitor alert versions for a specified monitor instance.

        Args:
            model_monitor_version (str): The unique ID associated with the model monitor.

        Returns:
            list[MonitorAlertVersion]: A list of monitor alert versions."""
        return self._call_api('listMonitorAlertVersionsForMonitorVersion', 'GET', query_params={'modelMonitorVersion': model_monitor_version}, parse_type=MonitorAlertVersion)

    def get_model_monitoring_logs(self, model_monitor_version: str, stdout: bool = False, stderr: bool = False) -> FunctionLogs:
        """Returns monitoring logs for the model.

        Args:
            model_monitor_version (str): The unique version ID of the model monitor version
            stdout (bool): Set True to get info logs
            stderr (bool): Set True to get error logs

        Returns:
            FunctionLogs: A function logs."""
        return self._call_api('getModelMonitoringLogs', 'GET', query_params={'modelMonitorVersion': model_monitor_version, 'stdout': stdout, 'stderr': stderr}, parse_type=FunctionLogs)

    def get_drift_for_feature(self, model_monitor_version: str, feature_name: str, nested_feature_name: str = None) -> FeatureDistribution:
        """Gets the feature drift associated with a single feature in an output feature group from a prediction.

        Args:
            model_monitor_version (str): Unique string identifier of a model monitor version created under the project.
            feature_name (str): Name of the feature to view the distribution of.
            nested_feature_name (str): Optionally, the name of the nested feature that the feature is in.

        Returns:
            FeatureDistribution: An object describing the training and prediction output feature distributions."""
        return self._call_api('getDriftForFeature', 'GET', query_params={'modelMonitorVersion': model_monitor_version, 'featureName': feature_name, 'nestedFeatureName': nested_feature_name}, parse_type=FeatureDistribution)

    def get_outliers_for_feature(self, model_monitor_version: str, feature_name: str = None, nested_feature_name: str = None) -> Dict:
        """Gets a list of outliers measured by a single feature (or overall) in an output feature group from a prediction.

        Args:
            model_monitor_version (str): Unique string identifier for a model monitor version created under the project.
            feature_name (str): Name of the feature to view the distribution of.
            nested_feature_name (str): Optionally, the name of the nested feature that the feature is in."""
        return self._call_api('getOutliersForFeature', 'GET', query_params={'modelMonitorVersion': model_monitor_version, 'featureName': feature_name, 'nestedFeatureName': nested_feature_name})

    def get_outliers_for_batch_prediction_feature(self, batch_prediction_version: str, feature_name: str = None, nested_feature_name: str = None) -> Dict:
        """Gets a list of outliers measured by a single feature (or overall) in an output feature group from a prediction.

        Args:
            batch_prediction_version (str): Unique string identifier for a batch prediction version created under the project.
            feature_name (str): Name of the feature to view the distribution of.
            nested_feature_name (str): Optionally, the name of the nested feature that the feature is in."""
        return self._call_api('getOutliersForBatchPredictionFeature', 'GET', query_params={'batchPredictionVersion': batch_prediction_version, 'featureName': feature_name, 'nestedFeatureName': nested_feature_name})

    def describe_deployment(self, deployment_id: str) -> Deployment:
        """Retrieves a full description of the specified deployment.

        Args:
            deployment_id (str): Unique string identifier associated with the deployment.

        Returns:
            Deployment: Description of the deployment."""
        return self._call_api('describeDeployment', 'GET', query_params={'deploymentId': deployment_id}, parse_type=Deployment)

    def list_deployments(self, project_id: str) -> List[Deployment]:
        """Retrieves a list of all deployments in the specified project.

        Args:
            project_id (str): The unique identifier associated with the project.

        Returns:
            list[Deployment]: An array of deployments."""
        return self._call_api('listDeployments', 'GET', query_params={'projectId': project_id}, parse_type=Deployment)

    def list_deployment_tokens(self, project_id: str) -> List[DeploymentAuthToken]:
        """Retrieves a list of all deployment tokens associated with the specified project.

        Args:
            project_id (str): The unique ID associated with the project.

        Returns:
            list[DeploymentAuthToken]: A list of deployment tokens."""
        return self._call_api('listDeploymentTokens', 'GET', query_params={'projectId': project_id}, parse_type=DeploymentAuthToken)

    def get_api_endpoint(self, deployment_token: str = None, deployment_id: str = None, streaming_token: str = None, feature_group_id: str = None, model_id: str = None) -> ApiEndpoint:
        """Returns the API endpoint specific to an organization. This function can be utilized using either an API Key or a deployment ID and token for authentication.

        Args:
            deployment_token (str): Token used for authenticating access to deployed models.
            deployment_id (str): Unique identifier assigned to a deployment created under the specified project.
            streaming_token (str): Token used for authenticating access to streaming data.
            feature_group_id (str): Unique identifier assigned to a feature group.
            model_id (str): Unique identifier assigned to a model.

        Returns:
            ApiEndpoint: The API endpoint specific to the organization."""
        return self._call_api('getApiEndpoint', 'GET', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id, 'streamingToken': streaming_token, 'featureGroupId': feature_group_id, 'modelId': model_id}, parse_type=ApiEndpoint)

    def get_model_training_types_for_deployment(self, model_id: str, model_version: str = None, algorithm: str = None) -> ModelTrainingTypeForDeployment:
        """Returns types of models that can be deployed for a given model instance ID.

        Args:
            model_id (str): The unique ID associated with the model.
            model_version (str): The unique ID associated with the model version to deploy.
            algorithm (str): The unique ID associated with the algorithm to deploy.

        Returns:
            ModelTrainingTypeForDeployment: Model training types for deployment."""
        return self._call_api('getModelTrainingTypesForDeployment', 'GET', query_params={'modelId': model_id, 'modelVersion': model_version, 'algorithm': algorithm}, parse_type=ModelTrainingTypeForDeployment)

    def describe_refresh_policy(self, refresh_policy_id: str) -> RefreshPolicy:
        """Retrieve a single refresh policy

        Args:
            refresh_policy_id (str): The unique ID associated with this refresh policy.

        Returns:
            RefreshPolicy: An object representing the refresh policy."""
        return self._call_api('describeRefreshPolicy', 'GET', query_params={'refreshPolicyId': refresh_policy_id}, parse_type=RefreshPolicy)

    def describe_refresh_pipeline_run(self, refresh_pipeline_run_id: str) -> RefreshPipelineRun:
        """Retrieve a single refresh pipeline run

        Args:
            refresh_pipeline_run_id (str): Unique string identifier associated with the refresh pipeline run.

        Returns:
            RefreshPipelineRun: A refresh pipeline run object."""
        return self._call_api('describeRefreshPipelineRun', 'GET', query_params={'refreshPipelineRunId': refresh_pipeline_run_id}, parse_type=RefreshPipelineRun)

    def list_refresh_policies(self, project_id: str = None, dataset_ids: list = [], feature_group_id: str = None, model_ids: list = [], deployment_ids: list = [], batch_prediction_ids: list = [], model_monitor_ids: list = [], prediction_metric_ids: list = [], notebook_ids: list = []) -> List[RefreshPolicy]:
        """List the refresh policies for the organization

        Args:
            project_id (str): Optionally, a Project ID can be specified so that all datasets, models, deployments, batch predictions, prediction metrics, model monitors, and notebooks are captured at the instant this policy was created.
            dataset_ids (list): Comma-separated list of Dataset IDs.
            feature_group_id (str): Feature Group ID for which we wish to see the refresh policies attached.
            model_ids (list): Comma-separated list of Model IDs.
            deployment_ids (list): Comma-separated list of Deployment IDs.
            batch_prediction_ids (list): Comma-separated list of Batch Prediction IDs.
            model_monitor_ids (list): Comma-separated list of Model Monitor IDs.
            prediction_metric_ids (list): Comma-separated list of Prediction Metric IDs.
            notebook_ids (list): Comma-separated list of Notebook IDs.

        Returns:
            list[RefreshPolicy]: List of all refresh policies in the organization."""
        return self._call_api('listRefreshPolicies', 'GET', query_params={'projectId': project_id, 'datasetIds': dataset_ids, 'featureGroupId': feature_group_id, 'modelIds': model_ids, 'deploymentIds': deployment_ids, 'batchPredictionIds': batch_prediction_ids, 'modelMonitorIds': model_monitor_ids, 'predictionMetricIds': prediction_metric_ids, 'notebookIds': notebook_ids}, parse_type=RefreshPolicy)

    def list_refresh_pipeline_runs(self, refresh_policy_id: str) -> List[RefreshPipelineRun]:
        """List the the times that the refresh policy has been run

        Args:
            refresh_policy_id (str): Unique identifier associated with the refresh policy.

        Returns:
            list[RefreshPipelineRun]: List of refresh pipeline runs for the given refresh policy ID."""
        return self._call_api('listRefreshPipelineRuns', 'GET', query_params={'refreshPolicyId': refresh_policy_id}, parse_type=RefreshPipelineRun)

    def download_batch_prediction_result_chunk(self, batch_prediction_version: str, offset: int = 0, chunk_size: int = 10485760) -> io.BytesIO:
        """Returns a stream containing the batch prediction results.

        Args:
            batch_prediction_version (str): Unique string identifier of the batch prediction version to get the results from.
            offset (int): The offset to read from.
            chunk_size (int): The maximum amount of data to read."""
        return self._call_api('downloadBatchPredictionResultChunk', 'GET', query_params={'batchPredictionVersion': batch_prediction_version, 'offset': offset, 'chunkSize': chunk_size}, streamable_response=True)

    def get_batch_prediction_connector_errors(self, batch_prediction_version: str) -> io.BytesIO:
        """Returns a stream containing the batch prediction database connection write errors, if any writes failed for the specified batch prediction job.

        Args:
            batch_prediction_version (str): Unique string identifier of the batch prediction job to get the errors for."""
        return self._call_api('getBatchPredictionConnectorErrors', 'GET', query_params={'batchPredictionVersion': batch_prediction_version}, streamable_response=True)

    def list_batch_predictions(self, project_id: str) -> List[BatchPrediction]:
        """Retrieves a list of batch predictions in the project.

        Args:
            project_id (str): Unique string identifier of the project.

        Returns:
            list[BatchPrediction]: List of batch prediction jobs."""
        return self._call_api('listBatchPredictions', 'GET', query_params={'projectId': project_id}, parse_type=BatchPrediction)

    def describe_batch_prediction(self, batch_prediction_id: str) -> BatchPrediction:
        """Describe the batch prediction.

        Args:
            batch_prediction_id (str): The unique identifier associated with the batch prediction.

        Returns:
            BatchPrediction: The batch prediction description."""
        return self._call_api('describeBatchPrediction', 'GET', query_params={'batchPredictionId': batch_prediction_id}, parse_type=BatchPrediction)

    def list_batch_prediction_versions(self, batch_prediction_id: str, limit: int = 100, start_after_version: str = None) -> List[BatchPredictionVersion]:
        """Retrieves a list of versions of a given batch prediction

        Args:
            batch_prediction_id (str): Unique identifier of the batch prediction.
            limit (int): Number of versions to list.
            start_after_version (str): Version to start after.

        Returns:
            list[BatchPredictionVersion]: List of batch prediction versions."""
        return self._call_api('listBatchPredictionVersions', 'GET', query_params={'batchPredictionId': batch_prediction_id, 'limit': limit, 'startAfterVersion': start_after_version}, parse_type=BatchPredictionVersion)

    def describe_batch_prediction_version(self, batch_prediction_version: str) -> BatchPredictionVersion:
        """Describes a Batch Prediction Version.

        Args:
            batch_prediction_version (str): Unique string identifier of the Batch Prediction Version.

        Returns:
            BatchPredictionVersion: The Batch Prediction Version."""
        return self._call_api('describeBatchPredictionVersion', 'GET', query_params={'batchPredictionVersion': batch_prediction_version}, parse_type=BatchPredictionVersion)

    def describe_python_function(self, name: str) -> PythonFunction:
        """Describe a Python Function.

        Args:
            name (str): The name to identify the Python function.

        Returns:
            PythonFunction: The Python function object."""
        return self._call_api('describePythonFunction', 'GET', query_params={'name': name}, parse_type=PythonFunction)

    def list_python_functions(self, function_type: str = 'FEATURE_GROUP') -> List[PythonFunction]:
        """List all python functions within the organization.

        Args:
            function_type (str): Optional argument to specify the type of function to list Python functions for; defaults to FEATURE_GROUP.

        Returns:
            list[PythonFunction]: A list of PythonFunction objects."""
        return self._call_api('listPythonFunctions', 'GET', query_params={'functionType': function_type}, parse_type=PythonFunction)

    def list_pipelines(self, project_id: str = None) -> List[Pipeline]:
        """Lists the pipelines for an organization or a project

        Args:
            project_id (str): Unique string identifier for the project to list graph dashboards from.

        Returns:
            list[Pipeline]: A list of pipelines."""
        return self._call_api('listPipelines', 'GET', query_params={'projectId': project_id}, parse_type=Pipeline)

    def describe_pipeline_version(self, pipeline_version: str) -> PipelineVersion:
        """Describes a specified pipeline version

        Args:
            pipeline_version (str): Unique string identifier for the pipeline version

        Returns:
            PipelineVersion: Object describing the pipeline version"""
        return self._call_api('describePipelineVersion', 'GET', query_params={'pipelineVersion': pipeline_version}, parse_type=PipelineVersion)

    def describe_pipeline_step(self, pipeline_step_id: str) -> PipelineStep:
        """Deletes a step from a pipeline.

        Args:
            pipeline_step_id (str): The ID of the pipeline step.

        Returns:
            PipelineStep: An object describing the pipeline step."""
        return self._call_api('describePipelineStep', 'GET', query_params={'pipelineStepId': pipeline_step_id}, parse_type=PipelineStep)

    def describe_pipeline_step_by_name(self, pipeline_id: str, step_name: str) -> PipelineStep:
        """Describes a pipeline step by the step name.

        Args:
            pipeline_id (str): The ID of the pipeline.
            step_name (str): The name of the step.

        Returns:
            PipelineStep: An object describing the pipeline step."""
        return self._call_api('describePipelineStepByName', 'GET', query_params={'pipelineId': pipeline_id, 'stepName': step_name}, parse_type=PipelineStep)

    def unset_pipeline_refresh_schedule(self, pipeline_id: str) -> Pipeline:
        """Deletes the refresh schedule for a given pipeline.

        Args:
            pipeline_id (str): The id of the pipeline.

        Returns:
            Pipeline: Object describing the pipeline."""
        return self._call_api('unsetPipelineRefreshSchedule', 'GET', query_params={'pipelineId': pipeline_id}, parse_type=Pipeline)

    def pause_pipeline_refresh_schedule(self, pipeline_id: str) -> Pipeline:
        """Pauses the refresh schedule for a given pipeline.

        Args:
            pipeline_id (str): The id of the pipeline.

        Returns:
            Pipeline: Object describing the pipeline."""
        return self._call_api('pausePipelineRefreshSchedule', 'GET', query_params={'pipelineId': pipeline_id}, parse_type=Pipeline)

    def resume_pipeline_refresh_schedule(self, pipeline_id: str) -> Pipeline:
        """Resumes the refresh schedule for a given pipeline.

        Args:
            pipeline_id (str): The id of the pipeline.

        Returns:
            Pipeline: Object describing the pipeline."""
        return self._call_api('resumePipelineRefreshSchedule', 'GET', query_params={'pipelineId': pipeline_id}, parse_type=Pipeline)

    def list_pipeline_version_logs(self, pipeline_version: str) -> PipelineVersionLogs:
        """Gets the logs for the steps in a given pipeline version.

        Args:
            pipeline_version (str): The id of the pipeline version.

        Returns:
            PipelineVersionLogs: Object describing the logs for the steps in the pipeline."""
        return self._call_api('listPipelineVersionLogs', 'GET', query_params={'pipelineVersion': pipeline_version}, parse_type=PipelineVersionLogs)

    def get_step_version_logs(self, pipeline_step_version: str) -> PipelineStepVersionLogs:
        """Gets the logs for a given step version.

        Args:
            pipeline_step_version (str): The id of the pipeline step version.

        Returns:
            PipelineStepVersionLogs: Object describing the pipeline step logs."""
        return self._call_api('getStepVersionLogs', 'GET', query_params={'pipelineStepVersion': pipeline_step_version}, parse_type=PipelineStepVersionLogs)

    def describe_graph_dashboard(self, graph_dashboard_id: str) -> GraphDashboard:
        """Describes a given graph dashboard.

        Args:
            graph_dashboard_id (str): Unique identifier for the graph dashboard.

        Returns:
            GraphDashboard: An object containing information about the graph dashboard."""
        return self._call_api('describeGraphDashboard', 'GET', query_params={'graphDashboardId': graph_dashboard_id}, parse_type=GraphDashboard)

    def list_graph_dashboards(self, project_id: str = None) -> List[GraphDashboard]:
        """Lists the graph dashboards for a project

        Args:
            project_id (str): Unique string identifier for the project to list graph dashboards from.

        Returns:
            list[GraphDashboard]: A list of graph dashboards."""
        return self._call_api('listGraphDashboards', 'GET', query_params={'projectId': project_id}, parse_type=GraphDashboard)

    def delete_graph_from_dashboard(self, graph_reference_id: str):
        """Deletes a python plot function from a dashboard

        Args:
            graph_reference_id (str): Unique String Identifier for the graph"""
        return self._call_api('deleteGraphFromDashboard', 'GET', query_params={'graphReferenceId': graph_reference_id})

    def describe_graph_for_dashboard(self, graph_reference_id: str) -> PythonPlotFunction:
        """Describes a python plot to a graph dashboard

        Args:
            graph_reference_id (str): Unique string identifier for the python function id for the graph

        Returns:
            PythonPlotFunction: An object describing the graph dashboard."""
        return self._call_api('describeGraphForDashboard', 'GET', query_params={'graphReferenceId': graph_reference_id}, parse_type=PythonPlotFunction)

    def describe_algorithm(self, algorithm: str) -> Algorithm:
        """Retrieves a full description of the specified algorithm.

        Args:
            algorithm (str): The name of the algorithm.

        Returns:
            Algorithm: The description of the algorithm."""
        return self._call_api('describeAlgorithm', 'GET', query_params={'algorithm': algorithm}, parse_type=Algorithm)

    def list_algorithms(self, problem_type: str = None, project_id: str = None) -> List[Algorithm]:
        """List all custom algorithms, with optional filtering on Problem Type and Project ID

        Args:
            problem_type (str): The problem type to query. If `None`, return all algorithms in the organization.
            project_id (str): The ID of the project.

        Returns:
            list[Algorithm]: A list of algorithms."""
        return self._call_api('listAlgorithms', 'GET', query_params={'problemType': problem_type, 'projectId': project_id}, parse_type=Algorithm)

    def describe_custom_loss_function(self, name: str) -> CustomLossFunction:
        """Retrieve a full description of a previously registered custom loss function.

        Args:
            name (str): Registered name of the custom loss function.

        Returns:
            CustomLossFunction: The description of the custom loss function with the given name."""
        return self._call_api('describeCustomLossFunction', 'GET', query_params={'name': name}, parse_type=CustomLossFunction)

    def list_custom_loss_functions(self, name_prefix: str = None, loss_function_type: str = None) -> CustomLossFunction:
        """Retrieves a list of registered custom loss functions and their descriptions.

        Args:
            name_prefix (str): The prefix of the names of the loss functions to list.
            loss_function_type (str): The category of loss functions to search in.

        Returns:
            CustomLossFunction: The description of the custom loss function with the given name."""
        return self._call_api('listCustomLossFunctions', 'GET', query_params={'namePrefix': name_prefix, 'lossFunctionType': loss_function_type}, parse_type=CustomLossFunction)

    def describe_custom_metric(self, name: str) -> CustomMetric:
        """Retrieves a full description of a previously registered custom metric function.

        Args:
            name (str): Registered name of the custom metric.

        Returns:
            CustomMetric: The description of the custom metric with the given name."""
        return self._call_api('describeCustomMetric', 'GET', query_params={'name': name}, parse_type=CustomMetric)

    def describe_custom_metric_version(self, custom_metric_version: str) -> CustomMetricVersion:
        """Describes a given custom metric version

        Args:
            custom_metric_version (str): A unique string identifier for the custom metric version.

        Returns:
            CustomMetricVersion: An object describing the custom metric version."""
        return self._call_api('describeCustomMetricVersion', 'GET', query_params={'customMetricVersion': custom_metric_version}, parse_type=CustomMetricVersion)

    def list_custom_metrics(self, name_prefix: str = None, problem_type: str = None) -> List[CustomMetric]:
        """Retrieves a list of registered custom metrics.

        Args:
            name_prefix (str): The prefix of the names of the custom metrics.
            problem_type (str): The associated problem type of the custom metrics.

        Returns:
            list[CustomMetric]: A list of custom metrics."""
        return self._call_api('listCustomMetrics', 'GET', query_params={'namePrefix': name_prefix, 'problemType': problem_type}, parse_type=CustomMetric)

    def describe_module(self, name: str) -> Module:
        """Retrieves a full description of the specified module.

        Args:
            name (str): The name of the module.

        Returns:
            Module: The description of the module."""
        return self._call_api('describeModule', 'GET', query_params={'name': name}, parse_type=Module)

    def list_modules(self) -> List[Module]:
        """List all the modules

        Returns:
            list[Module]: A list of modules"""
        return self._call_api('listModules', 'GET', query_params={}, parse_type=Module)

    def get_natural_language_explanation(self, feature_group_id: str = None, feature_group_version: str = None, model_id: str = None) -> NaturalLanguageExplanation:
        """Returns the saved natural language explanation of an artifact with given ID. The artifact can be - Feature Group or Feature Group Version or Model

        Args:
            feature_group_id (str): A unique string identifier associated with the Feature Group.
            feature_group_version (str): A unique string identifier associated with the Feature Group Version.
            model_id (str): A unique string identifier associated with the Model.

        Returns:
            NaturalLanguageExplanation: The object containing natural language explanation(s) as field(s)."""
        return self._call_api('getNaturalLanguageExplanation', 'GET', query_params={'featureGroupId': feature_group_id, 'featureGroupVersion': feature_group_version, 'modelId': model_id}, parse_type=NaturalLanguageExplanation)

    def generate_natural_language_explanation(self, feature_group_id: str = None, feature_group_version: str = None, model_id: str = None) -> NaturalLanguageExplanation:
        """Generates natural language explanation of an artifact with given ID. The artifact can be - Feature Group or Feature Group Version or Model

        Args:
            feature_group_id (str): A unique string identifier associated with the Feature Group.
            feature_group_version (str): A unique string identifier associated with the Feature Group Version.
            model_id (str): A unique string identifier associated with the Model.

        Returns:
            NaturalLanguageExplanation: The object containing natural language explanation(s) as field(s)."""
        return self._call_api('generateNaturalLanguageExplanation', 'GET', query_params={'featureGroupId': feature_group_id, 'featureGroupVersion': feature_group_version, 'modelId': model_id}, parse_type=NaturalLanguageExplanation)

    def get_chat_session(self, chat_session_id: str) -> ChatSession:
        """Gets a chat session from Abacus Chat.

        Args:
            chat_session_id (str): The chat session id

        Returns:
            ChatSession: The chat session with Abacus Chat"""
        return self._call_api('getChatSession', 'GET', query_params={'chatSessionId': chat_session_id}, parse_type=ChatSession)

    def list_chat_sessions(self, most_recent_per_project: bool = False) -> ChatSession:
        """Lists all chat sessions for the current user

        Args:
            most_recent_per_project (bool): An optional parameter whether to only return the most recent chat session per project. Default False.

        Returns:
            ChatSession: The chat sessions with Abacus Chat"""
        return self._call_api('listChatSessions', 'GET', query_params={'mostRecentPerProject': most_recent_per_project}, parse_type=ChatSession)

    def get_deployment_conversation(self, deployment_conversation_id: str) -> DeploymentConversation:
        """Gets a deployment conversation.

        Args:
            deployment_conversation_id (str): Unique ID of the conversation.

        Returns:
            DeploymentConversation: The deployment conversation."""
        return self._call_api('getDeploymentConversation', 'GET', query_params={'deploymentConversationId': deployment_conversation_id}, parse_type=DeploymentConversation)

    def list_deployment_conversations(self, deployment_id: str) -> List[DeploymentConversation]:
        """Lists all conversations for the given deployment and current user.

        Args:
            deployment_id (str): The deployment to get conversations for.

        Returns:
            list[DeploymentConversation]: The deployment conversations."""
        return self._call_api('listDeploymentConversations', 'GET', query_params={'deploymentId': deployment_id}, parse_type=DeploymentConversation)

    def search_feature_groups(self, text: str, num_results: int = 10, project_id: str = None, feature_group_ids: list = None) -> List[OrganizationSearchResult]:
        """Search feature groups based on text and filters.

        Args:
            text (str): Text to use for approximately matching feature groups.
            num_results (int): The maximum number of search results to retrieve. The length of the returned list is less than or equal to num_results.
            project_id (str): The ID of the project in which to restrict the search, if specified.
            feature_group_ids (list): A list of feagure group IDs to restrict the search to.

        Returns:
            list[OrganizationSearchResult]: A list of search results, each containing the retrieved object and its relevance score"""
        return self._call_api('searchFeatureGroups', 'GET', query_params={'text': text, 'numResults': num_results, 'projectId': project_id, 'featureGroupIds': feature_group_ids}, parse_type=OrganizationSearchResult)

    def render_feature_group_data_for_llm(self, feature_group_id: str, token_budget: int = None, render_format: str = 'markdown') -> LlmInput:
        """Encode feature groups as language model inputs.

        Args:
            feature_group_id (str): The ID of the feature groups to be encoded.
            token_budget (int): Enforce a given budget for each encoded feature group.
            render_format (str): One of `['markdown', 'json']`

        Returns:
            LlmInput: LLM input object comprising of information about the feature group with given ID."""
        return self._call_api('renderFeatureGroupDataForLLM', 'GET', query_params={'featureGroupId': feature_group_id, 'tokenBudget': token_budget, 'renderFormat': render_format}, parse_type=LlmInput)

    def query_feature_group_code_generator(self, query: str, language: str, project_id: str = None, llm_name: str = None, max_tokens: int = None) -> LlmResponse:
        """Send a query to the feature group code generator tool to generate code for the query.

        Args:
            query (str): A natural language query which specifies what the user wants out of the feature group or its code.
            language (str): The language in which code is to be generated. One of 'sql' or 'python'.
            project_id (str): A unique string identifier of the project in context of which the query is.
            llm_name (str): Name of the underlying LLM to be used for generation. Should be one of 'gpt-4' or 'gpt-3.5-turbo'. Default is auto selection.
            max_tokens (int): Maximum number of tokens to generate. If set, the model will just stop generating after this token limit is reached.

        Returns:
            LlmResponse: The response from the model, raw text and parsed components."""
        return self._call_api('queryFeatureGroupCodeGenerator', 'GET', query_params={'query': query, 'language': language, 'projectId': project_id, 'llmName': llm_name, 'maxTokens': max_tokens}, parse_type=LlmResponse, timeout=300)

    def list_document_retrievers(self, project_id: str, limit: int = 100, start_after_id: str = None) -> List[DocumentRetriever]:
        """List all the document retrievers.

        Args:
            project_id (str): The ID of project that the document retriever is created in.
            limit (int): The number of document retrievers to return.
            start_after_id (str): An offset parameter to exclude all document retrievers up to this specified ID.

        Returns:
            list[DocumentRetriever]: All the document retrievers in the organization associated with the specified project."""
        return self._call_api('listDocumentRetrievers', 'GET', query_params={'projectId': project_id, 'limit': limit, 'startAfterId': start_after_id}, parse_type=DocumentRetriever)

    def describe_document_retriever(self, document_retriever_id: str) -> DocumentRetriever:
        """Describe a Document Retriever.

        Args:
            document_retriever_id (str): A unique string identifier associated with the document retriever.

        Returns:
            DocumentRetriever: The document retriever object."""
        return self._call_api('describeDocumentRetriever', 'GET', query_params={'documentRetrieverId': document_retriever_id}, parse_type=DocumentRetriever)

    def describe_document_retriever_by_name(self, name: str) -> DocumentRetriever:
        """Describe a document retriever by its name.

        Args:
            name (str): The unique name of the document retriever to look up.

        Returns:
            DocumentRetriever: The Document Retriever."""
        return self._call_api('describeDocumentRetrieverByName', 'GET', query_params={'name': name}, parse_type=DocumentRetriever)

    def list_document_retriever_versions(self, document_retriever_id: str, limit: int = 100, start_after_version: str = None) -> List[DocumentRetrieverVersion]:
        """List all the document retriever versions with a given ID.

        Args:
            document_retriever_id (str): A unique string identifier associated with the document retriever.
            limit (int): The number of vector store versions to retrieve.
            start_after_version (str): An offset parameter to exclude all document retriever versions up to this specified one.

        Returns:
            list[DocumentRetrieverVersion]: All the document retriever versions associated with the document retriever."""
        return self._call_api('listDocumentRetrieverVersions', 'GET', query_params={'documentRetrieverId': document_retriever_id, 'limit': limit, 'startAfterVersion': start_after_version}, parse_type=DocumentRetrieverVersion)

    def describe_document_retriever_version(self, document_retriever_version: str) -> DocumentRetrieverVersion:
        """Describe a document retriever version.

        Args:
            document_retriever_version (str): A unique string identifier associated with the document retriever version.

        Returns:
            DocumentRetrieverVersion: The document retriever version object."""
        return self._call_api('describeDocumentRetrieverVersion', 'GET', query_params={'documentRetrieverVersion': document_retriever_version}, parse_type=DocumentRetrieverVersion)


def get_source_code_info(train_function: callable, predict_function: callable = None, predict_many_function: callable = None, initialize_function: callable = None, common_functions: list = None):
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
    if common_functions:
        for func in common_functions:
            function_source_code = function_source_code + \
                '\n\n' + inspect.getsource(func)
    return function_source_code, train_function.__name__, predict_function_name, predict_many_function_name, initialize_function_name


def get_module_code_from_notebook(file_path):
    from nbformat import NO_CONVERT, read

    nb_full_path = os.path.join(os.getcwd(), file_path)
    with open(nb_full_path) as fp:
        notebook = read(fp, NO_CONVERT)
    notebook['cells']
    source_code = None
    for c in notebook['cells']:
        if not c['cell_type'] == 'code':
            continue
        if not source_code and '#module_start#' in c['source']:
            source_code = c['source']
        elif source_code:
            source_code = source_code + '\n\n' + c['source']
        if '#module_end#' in c['source']:
            break
    return source_code


def include_modules(source_code, included_modules):
    if not source_code or not included_modules:
        return source_code
    return '\n'.join([f'from {m} import *' for m in included_modules]) + '\n\n' + source_code


class ApiClient(ReadOnlyClient):
    """
    Abacus.AI API Client

    Args:
        api_key (str): The api key to use as authentication to the server
        server (str): The base server url to use to send API requets to
        client_options (ClientOptions): Optional API client configurations
        skip_version_check (bool): If true, will skip checking the server's current API version on initializing the client
    """

    def create_dataset_from_pandas(self, feature_group_table_name: str, df: pd.DataFrame, clean_column_names: bool = False) -> Dataset:
        """
        [Deprecated]
        Creates a Dataset from a pandas dataframe

        Args:
            feature_group_table_name (str): The table name to assign to the feature group created by this call
            df (pandas.DataFrame): The dataframe to upload
            clean_column_names (bool): If true, the dataframe's column names will be automatically cleaned to be complaint with Abacus.AI's column requirements. Otherwise it will raise a ValueError.

        Returns:
            Dataset: The dataset object created
        """
        df = self._validate_pandas_df(df, clean_column_names)
        upload = self.create_dataset_from_upload(
            table_name=feature_group_table_name, file_format='PARQUET')
        return self._upload_from_pandas(upload, df)

    def create_dataset_version_from_pandas(self, table_name_or_id: str, df: pd.DataFrame, clean_column_names: bool = False) -> Dataset:
        """
        [Deprecated]
        Updates an existing dataset from a pandas dataframe

        Args:
            table_name_or_id (str): The table name of the feature group or the ID of the dataset to update
            df (pandas.DataFrame): The dataframe to upload
            clean_column_names (bool): If true, the dataframe's column names will be automatically cleaned to be complaint with Abacus.AI's column requirements. Otherwise it will raise a ValueError.

        Returns:
            Dataset: The dataset updated
        """
        df = self._validate_pandas_df(df, clean_column_names)
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
        return self._upload_from_pandas(upload, df)

    def create_feature_group_from_pandas_df(self, table_name: str, df, clean_column_names: bool = False) -> FeatureGroup:
        """Create a Feature Group from a local Pandas DataFrame.

        Args:
            table_name (str): The table name to assign to the feature group created by this call
            df (pandas.DataFrame): The dataframe to upload
            clean_column_names (bool): If true, the dataframe's column names will be automatically cleaned to be complaint with Abacus.AI's column requirements. Otherwise it will raise a ValueError.
        """
        df = self._validate_pandas_df(df, clean_column_names)
        dataset = self.create_dataset_from_pandas(
            feature_group_table_name=table_name, df=df)
        return dataset.describe_feature_group()

    def update_feature_group_from_pandas_df(self, table_name: str, df, clean_column_names: bool = False) -> FeatureGroup:
        """Updates a DATASET Feature Group from a local Pandas DataFrame.

        Args:
            table_name (str): The table name to assign to the feature group created by this call
            df (pandas.DataFrame): The dataframe to upload
            clean_column_names (bool): If true, the dataframe's column names will be automatically cleaned to be complaint with Abacus.AI's column requirements. Otherwise it will raise a ValueError.
        """
        df = self._validate_pandas_df(df, clean_column_names)
        feature_group = self.describe_feature_group_by_table_name(table_name)
        if feature_group.feature_group_source_type != 'DATASET':
            raise ApiException(
                'Feature Group is not source type DATASET', 409, 'ConflictError')
        dataset_id = feature_group.dataset_id
        upload = self.create_dataset_version_from_upload(
            dataset_id, file_format='PARQUET')
        return self._upload_from_pandas(upload, df).describe_feature_group()

    def create_feature_group_from_spark_df(self, table_name: str, df) -> FeatureGroup:
        """Create a Feature Group from a local Spark DataFrame.

        Args:
            df (pyspark.sql.DataFrame): The dataframe to upload
            table_name (str): The table name to assign to the feature group created by this call
        """

        upload = self.create_dataset_from_upload(
            table_name=table_name, file_format='TAR')
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

    def create_model_from_functions(self, project_id: str, train_function: callable, predict_function: callable = None, training_input_tables: list = None, predict_many_function: callable = None, initialize_function: callable = None, cpu_size: str = None, memory: int = None, training_config: dict = None, exclusive_run: bool = False, included_modules: list = None, package_requirements: list = None, name: str = None, use_gpu: bool = False):
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
            package_requirements (List): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            included_modules (list): A list of user-created modules that will be included, which is equivalent to 'from module import *'
            name (str): The name of the model
            use_gpu (bool): Whether this model needs gpu
        """
        function_source_code, train_function_name, predict_function_name, predict_many_function_name, initialize_function_name = get_source_code_info(
            train_function, predict_function, predict_many_function, initialize_function)
        function_source_code = include_modules(
            function_source_code, included_modules)
        return self.create_model_from_python(project_id=project_id, function_source_code=function_source_code, train_function_name=train_function_name, predict_function_name=predict_function_name, predict_many_function_name=predict_many_function_name, initialize_function_name=initialize_function_name, training_input_tables=training_input_tables, training_config=training_config, cpu_size=cpu_size, memory=memory, exclusive_run=exclusive_run, package_requirements=package_requirements, name=name, use_gpu=use_gpu)

    def update_model_from_functions(self, model_id: str, train_function: callable, predict_function: callable = None, predict_many_function: callable = None, initialize_function: callable = None, training_input_tables: list = None, cpu_size: str = None, memory: int = None, included_modules: list = None, package_requirements: list = None, use_gpu: bool = False):
        """
        Creates a model from a python function. Please pass in all the functions, even if you don't update it.

        Args:
            model_id (str): The id of the model to update
            train_function (callable): The training fucntion callable to serialize and upload
            predict_function (callable): The predict function callable to serialize and upload
            predict_many_function (callable): The predict many function callable to serialize and upload
            initialize_function (callable): The initialize function callable to serialize and upload
            training_input_tables (list): The input table names of the feature groups to pass to the train function
            cpu_size (str): Size of the cpu for the training function
            memory (int): Memory (in GB) for the training function
            package_requirements (List): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            included_modules (list): A list of user-created modules that will be included, which is equivalent to 'from module import *'
            use_gpu (bool): Whether this model needs gpu
        """
        function_source_code, train_function_name, predict_function_name, predict_many_function_name, initialize_function_name = get_source_code_info(
            train_function, predict_function, predict_many_function, initialize_function)
        function_source_code = include_modules(
            function_source_code, included_modules)
        return self.update_python_model(model_id=model_id, function_source_code=function_source_code, train_function_name=train_function_name, predict_function_name=predict_function_name, predict_many_function_name=predict_many_function_name, initialize_function_name=initialize_function_name, training_input_tables=training_input_tables, cpu_size=cpu_size, memory=memory, package_requirements=package_requirements, use_gpu=use_gpu)

    def create_pipeline_step_from_function(self,
                                           pipeline_id: str,
                                           step_name: str,
                                           function: callable,
                                           step_input_mappings: list = None,
                                           output_variable_mappings: list = None,
                                           step_dependencies: list = None,
                                           package_requirements: list = None,
                                           cpu_size: str = None,
                                           memory: int = None):
        """
        Creates a step in a given pipeline from a python function.

        Args:
            pipeline_id (str): The ID of the pipeline to add the step to.
            step_name (str): The name of the step.
            function (callable): The python function.
            step_input_mappings (List[PythonFunctionArguments]): List of Python function arguments.
            output_variable_mappings (List[OutputVariableMapping]): List of Python function ouputs.
            step_dependencies (List[str]): List of step names this step depends on.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            cpu_size (str): Size of the CPU for the step function.
            memory (int): Memory (in GB) for the step function.

        """
        source_code = inspect.getsource(function)
        return self.create_pipeline_step(pipeline_id=pipeline_id,
                                         step_name=step_name,
                                         function_name=function.__name__,
                                         source_code=source_code,
                                         step_input_mappings=step_input_mappings,
                                         output_variable_mappings=output_variable_mappings,
                                         step_dependencies=step_dependencies,
                                         package_requirements=package_requirements,
                                         cpu_size=cpu_size,
                                         memory=memory)

    def update_pipeline_step_from_function(self,
                                           pipeline_step_id: str,
                                           function: callable,
                                           step_input_mappings: list,
                                           output_variable_mappings: list = None,
                                           step_dependencies: list = None,
                                           package_requirements: list = None,
                                           cpu_size: str = None,
                                           memory: int = None):
        """
        Updates a pipeline step from a python function.

        Args:
            pipeline_step_id (str): The ID of the pipeline_step to update.
            function (callable): The python function.
            step_input_mappings (List[PythonFunctionArguments]): List of Python function arguments.
            output_variable_mappings (List[OutputVariableMapping]): List of Python function ouputs.
            step_dependencies (List[str]): List of step names this step depends on.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            cpu_size (str): Size of the CPU for the step function.
            memory (int): Memory (in GB) for the step function.

        """
        source_code = inspect.getsource(function)

        return self.update_pipeline_step(pipeline_step_id=pipeline_step_id,
                                         function_name=function.__name__,
                                         source_code=source_code,
                                         step_input_mappings=step_input_mappings,
                                         output_variable_mappings=output_variable_mappings,
                                         step_dependencies=step_dependencies,
                                         cpu_size=cpu_size,
                                         memory=memory)

    def create_feature_group_from_python_function(self, function: callable, table_name: str, input_tables: list = None, python_function_name: str = None, python_function_bindings: list = None, cpu_size: str = None, memory: int = None, package_requirements: list = None, included_modules: list = None):
        """
        Creates a feature group from a python function

        Args:
            function (callable): The function callable for the feature group
            table_name (str): The table name to give the feature group
            input_tables (list): The input table names of the feature groups as input to the feature group function
            python_function_name (str): The name of the python function to create a feature group from.
            python_function_bindings (List<PythonFunctionArguments>): List of python function arguments
            cpu_size (str): Size of the cpu for the feature group function
            memory (int): Memory (in GB) for the feature group function
            package_requirements (List): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            included_modules (list): A list of user-created modules that will be included, which is equivalent to 'from module import *'
        """
        if function:
            function_source = get_clean_function_source_code(function)
            function_source = include_modules(
                function_source, included_modules)
            python_function_name = python_function_name or function.__name__
            python_function = self.create_python_function(name=python_function_name, source_code=function_source, function_name=function.__name__,
                                                          package_requirements=package_requirements, function_variable_mappings=python_function_bindings)
            if not python_function_bindings and input_tables:
                python_function_bindings = []
                for index, feature_group_table_name in enumerate(input_tables):
                    variable = python_function.function_variable_mappings[index]
                    python_function_bindings.append(
                        {'name': variable['name'], 'variable_type': variable['variable_type'], 'value': feature_group_table_name})
        return self.create_feature_group_from_function(table_name=table_name, python_function_name=python_function_name, python_function_bindings=python_function_bindings, cpu_size=cpu_size, memory=memory)

    def update_python_function_code(self, name: str, function: callable = None, function_variable_mappings: list = None, package_requirements: list = None, included_modules: list = None):
        """
        Update custom python function with user inputs for the given python function.

        Args:
            name (String): The unique name to identify the python function in an organization.
            function (callable): The function callable to serialize and upload.
            function_variable_mappings (List<PythonFunctionArguments>): List of python function arguments
            package_requirements (List): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            included_modules (list): A list of user-created modules that will be included, which is equivalent to 'from module import *'
        Returns:
            PythonFunction: The python_function object.
        """
        source_code = inspect.getsource(function)
        source_code = include_modules(source_code, included_modules)
        return self.update_python_function(name=name, source_code=source_code, function_name=function.__name__, function_variable_mappings=function_variable_mappings, package_requirements=package_requirements)

    def create_algorithm_from_function(self, name: str, problem_type: str, training_data_parameter_names_mapping: dict = None, training_config_parameter_name: str = None, train_function: callable = None, predict_function: callable = None, predict_many_function: callable = None, initialize_function: callable = None, common_functions: list = None, config_options: dict = None, is_default_enabled: bool = False, project_id: str = None, use_gpu: bool = False, package_requirements: list = None, included_modules: list = None):
        """
        Create a new algorithm, or update existing algorithm if the name already exists

        Args:
            name (String): The name to identify the algorithm, only uppercase letters, numbers and underscore allowed
            problem_type (str): The type of the problem this algorithm will work on
            train_function (callable): The training function callable to serialize and upload
            predict_function (callable): The predict function callable to serialize and upload
            predict_many_function (callable): The predict many function callable to serialize and upload
            initialize_function (callable): The initialize function callable to serialize and upload
            common_functions (List of callables): A list of functions that will be used by both train and predict functions, e.g. some data processing utilities
            training_data_parameter_names_mapping (Dict): The mapping from feature group types to training data parameter names in the train function
            training_config_parameter_name (string): The train config parameter name in the train function
            config_options (Dict): Map dataset types and configs to train function parameter names
            is_default_enabled: Whether train with the algorithm by default
            project_id (Unique String Identifier): The unique version ID of the project
            use_gpu (Boolean): Whether this algorithm needs to run on GPU
            package_requirements (List): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            included_modules (list): A list of user-created modules that will be included, which is equivalent to 'from module import *'
        """
        source_code, train_function_name, predict_function_name, predict_many_function_name, initialize_function_name = get_source_code_info(
            train_function, predict_function, predict_many_function, initialize_function, common_functions)
        source_code = include_modules(source_code, included_modules)
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
            project_id=project_id,
            use_gpu=use_gpu,
            package_requirements=package_requirements)

    def update_algorithm_from_function(self, algorithm: str, training_data_parameter_names_mapping: dict = None, training_config_parameter_name: str = None, train_function: callable = None, predict_function: callable = None, predict_many_function: callable = None, initialize_function: callable = None, common_functions: list = None, config_options: dict = None, is_default_enabled: bool = None, use_gpu: bool = None, package_requirements: list = None, included_modules: list = None):
        """
        Create a new algorithm, or update existing algorithm if the name already exists

        Args:
            algorithm (String): The name to identify the algorithm, only uppercase letters, numbers and underscore allowed
            train_function (callable): The training fucntion callable to serialize and upload
            predict_function (callable): The predict function callable to serialize and upload
            predict_many_function (callable): The predict many function callable to serialize and upload
            initialize_function (callable): The initialize function callable to serialize and upload
            common_functions (List of callables): A list of functions that will be used by both train and predict functions, e.g. some data processing utilities
            training_data_parameter_names_mapping (Dict): The mapping from feature group types to training data parameter names in the train function
            training_config_parameter_name (string): The train config parameter name in the train function
            config_options (Dict): Map dataset types and configs to train function parameter names
            is_default_enabled (Boolean): Whether train with the algorithm by default
            use_gpu (Boolean): Whether this algorithm needs to run on GPU
            package_requirements (List): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            included_modules (list): A list of user-created modules that will be included, which is equivalent to 'from module import *'

        """
        source_code, train_function_name, predict_function_name, predict_many_function_name, initialize_function_name = get_source_code_info(
            train_function, predict_function, predict_many_function, initialize_function, common_functions)
        source_code = include_modules(source_code, included_modules)
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
            is_default_enabled=is_default_enabled,
            use_gpu=use_gpu,
            package_requirements=package_requirements)

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

    def get_train_function_input_from_model_version(self, model_version: str, algorithm: str = None, training_config: dict = None, custom_algorithm_config: any = None):
        """
        Get the input data for the train function to test locally, based on a trained model version.

        Args:
            model_version (String): The string identifier of the model version
            algorithm (String):  The particular algorithm's name, whose train function to test with
            training_config (Dict): A dictionary for Abacus.AI defined training options and values
            custom_algorithm_config (Any): User-defined config that can be serialized by JSON
        Returns:
            A dictionary that maps train function parameter names to their values.
        """

        model_version = self.describe_model_version(model_version)
        model = self.describe_model(model_version.model_id)
        project_id = model.project_id
        training_table_names = model.training_input_tables

        if not algorithm:
            deployable_algos = [
                algo['name'] for algo in model_version.deployable_algorithms if algo['name'].startswith('USER.')]
            algorithm = deployable_algos[0] if deployable_algos else None
        if not algorithm:
            raise ApiException(
                'Please provide explicit algorithm name, can not resolve automatically', 400, 'InvalidParameterError')

        algo_obj = self.describe_algorithm(algorithm)
        training_data_parameter_name_override = algo_obj.training_input_mappings.get(
            'training_data_parameter_names_mapping')
        training_config_parameter_name_override = algo_obj.training_input_mappings.get(
            'training_config_parameter_name')
        training_config = training_config or model_version.model_config
        custom_algorithm_config = custom_algorithm_config or model_version.custom_algorithm_configs.get(
            algorithm)

        train_function_info = self.get_custom_train_function_info(project_id=project_id, feature_group_names_for_training=training_table_names,
                                                                  training_data_parameter_name_override=training_data_parameter_name_override, training_config=training_config, custom_algorithm_config=custom_algorithm_config)
        train_data_parameter_to_feature_group_ids = train_function_info.train_data_parameter_to_feature_group_ids

        input = {parameter_name: self.describe_feature_group(fgid).load_as_pandas(
        ) for parameter_name, fgid in train_data_parameter_to_feature_group_ids.items()}
        input['schema_mappings'] = train_function_info.schema_mappings
        training_config_parameter_name = training_config_parameter_name_override or 'training_config'
        input[training_config_parameter_name] = train_function_info.training_config
        return input

    def create_custom_loss_function(self, name: str, loss_function_type: str, loss_function: Callable):
        """
        Registers a new custom loss function which can be used as an objective function during model training.

        Args:
            name (String): A name for the loss. Should be unique per organization. Limit - 50 chars. Only underscores, numbers, uppercase alphabets allowed
            loss_function_type (String): The category of problems that this loss would be applicable to. Ex - REGRESSION_DL_TF, CLASSIFICATION_DL_TF, etc.
            loss_function (Callable): A python functor which can take required arguments (Ex - (y_true, y_pred)) and returns loss value(s) (Ex - An array of loss values of size batch size)

        Returns:
            CustomLossFunction: A description of the registered custom loss function

        Raises:
            InvalidParameterError: If either loss function name or type or the passed function is invalid/incompatible
            AlreadyExistsError: If the loss function with the same name already exists in the organization
        """
        loss_function_name = loss_function.__name__
        loss_function_source_code = get_clean_function_source_code(
            loss_function)
        # Register the loss function
        clf = self.create_custom_loss_function_with_source_code(
            name, loss_function_type, loss_function_name, loss_function_source_code)
        return clf

    def update_custom_loss_function(self, name: str, loss_function: Callable):
        """
        Updates a previously registered custom loss function with a new function implementation.

        Args:
            name (String): name of the registered custom loss.
            loss_function (Callable): A python functor which can take required arguments (Ex - (y_true, y_pred)) and returns loss value(s) (Ex - An array of loss values of size batch size)

        Returns:
            CustomLossFunction: A description of the updated custom loss function

        Raises:
            InvalidParameterError: If either loss function name or type or the passed function is invalid/incompatible
            DataNotFoundError: If a loss function with given name is not found in the organization
        """
        loss_function_name = loss_function.__name__
        loss_function_source_code = get_clean_function_source_code(
            loss_function)
        # Register the loss function
        clf = self.update_custom_loss_function_with_source_code(
            name, loss_function_name, loss_function_source_code)
        return clf

    def create_custom_metric_from_function(self, name: str, problem_type: str, custom_metric_function: Callable):
        """
        Registers a new custom metric which can be used as an evaluation metric for the trained model.

        Args:
            name (String): A name for the metric. Should be unique per organization. Limit - 50 chars. Only underscores, numbers, uppercase alphabets allowed.
            problem_type (String): The problem type that this metric would be applicable to. e.g. - REGRESSION, FORECASTING, etc.
            custom_metric_function (Callable): A python functor which can take required arguments e.g. (y_true, y_pred) and returns the metric value.

        Returns:
            CustomMetric: The newly created custom metric.

        Raises:
            InvalidParameterError: If either custom metric name or type or the passed function is invalid/incompatible.
            AlreadyExistsError: If a custom metric with given name already exists in the organization.
        """
        custom_metric_function_name = custom_metric_function.__name__
        source_code = get_clean_function_source_code(custom_metric_function)

        # Register the custom metric
        custom_metric = self.create_custom_metric(
            name, problem_type, custom_metric_function_name, source_code)
        return custom_metric

    def update_custom_metric_from_function(self, name: str, custom_metric_function: Callable):
        """
        Updates a previously registered custom metric.

        Args:
            name (String): A name for the metric. Should be unique per organization. Limit - 50 chars. Only underscores, numbers, uppercase alphabets allowed.
            custom_metric_function (Callable): A python functor which can take required arguments e.g. (y_true, y_pred) and returns the metric value.

        Returns:
            CustomMetric: The updated custom metric.

        Raises:
            InvalidParameterError: If either custom metric name or type or the passed function is invalid/incompatible.
            DataNotFoundError: If a custom metric with given name is not found in the organization.
        """
        custom_metric_function_name = custom_metric_function.__name__
        source_code = get_clean_function_source_code(custom_metric_function)

        custom_metric = self.update_custom_metric(
            name, custom_metric_function_name, source_code)
        return custom_metric

    def create_module_from_notebook(self, file_path, name):
        """
        Create a module with the code marked in the notebook. Use '#module_start#' to mark the starting code cell and '#module_end#' for the
        ending code cell.

        Args:
            file_path (String): Notebook's relative path to the root directory, e.g. 'n1.ipynb'
            name (String): Name of the module to create.

        Returns:
            Module: the created Abacus.ai module object
        """
        source_code = get_module_code_from_notebook(file_path)
        return self.create_module(name=name, source_code=source_code)

    def update_module_from_notebook(self, file_path, name):
        """
        Update the module with the code marked in the notebook. Use '#module_start#' to mark the starting code cell and '#module_end#' for the
        ending code cell.

        Args:
            file_path (String):  Notebook's relative path to the root directory, e.g. 'n1.ipynb'
            name (String): Name of the module to create.

        Returns:
            Module: the created Abacus.ai module object
        """
        source_code = get_module_code_from_notebook(file_path)
        return self.update_module(name=name, source_code=source_code)

    def import_module(self, name):
        """
        Import a module created previously. It will reload if has been imported before.
        This will be equivalent to including from that module file.

        Args:
            name (String): Name of the module to import.

        Returns:
            module: the imported python module
        """
        module = self.describe_module(name)
        temp_dir = tempfile.gettempdir()
        with open(os.path.join(temp_dir, name + '.py'), 'w') as file:
            file.write(module.code_source.source_code)
        if temp_dir not in sys.path:
            sys.path.insert(0, temp_dir)
        import importlib
        if name in sys.modules:
            module = importlib.reload(sys.modules[name])
        else:
            module = importlib.import_module(name)

        # respect __all__ if exists
        if '__all__' in module.__dict__:
            names = module.__dict__['__all__']
        else:
            # otherwise we import all names that don't begin with _
            names = [x for x in module.__dict__ if not x.startswith('_')]
        import __main__ as the_main
        for name in names:
            setattr(the_main, name, getattr(module, name))
        return module

    def create_agent_from_function(self, project_id: str, agent_function: callable, name: str = None, memory: int = None, package_requirements: list = None):
        """
        Creates the agent from a python function

        Args:
            project_id (str): The project to create the model in
            agent_function (callable): The agent function callable to serialize and upload
            name (str): The name of the agent
            memory (int): Memory (in GB) for hosting the agent
            package_requirements (List): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
        """
        function_source_code = get_clean_function_source_code(agent_function)
        agent_function_name = agent_function.__name__
        return self.create_agent(project_id=project_id, function_source_code=function_source_code, agent_function_name=agent_function_name, name=name, memory=memory, package_requirements=package_requirements)

    def update_agent_with_function(self, model_id: str, agent_function: callable, memory: int = None, package_requirements: list = None):
        """
        Updates the agent with a new agent function.

        Args:
            model_id (str): The unique ID associated with the AI Agent to be changed.
            agent_function (callable): The new agent function callable to serialize and upload
            memory (int): Memory (in GB) for hosting the agent
            package_requirements (List): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
        """
        function_source_code = inspect.getsource(agent_function)
        agent_function_name = agent_function.__name__
        return self.update_agent(model_id=model_id, function_source_code=function_source_code, agent_function_name=agent_function_name, memory=memory, package_requirements=package_requirements)

    def execute_feature_group_sql(self, sql, timeout=3600, delay=2):
        """
        Execute a SQL query on the feature groups

        Args:
            sql (str): The SQL query to execute.

        Returns:
            pandas.DataFrame: The result of the query.
        """
        execute_query = self.execute_async_feature_group_operation(sql)
        execute_query.wait_for_execution(timeout=timeout, delay=delay)
        return execute_query.load_as_pandas()

    def get_agent_context_chat_history(self):
        """
        Gets a history of chat messages from the current request context. Applicable within a AIAgent
        execute function.

        Returns:
            List[ChatMessage]:: The chat history for the current request being processed by the Agent.
        """
        from .chat_message import ChatMessage
        return get_object_from_context(self, _request_context, 'chat_history', List[ChatMessage])

    def set_agent_context_chat_history(self, chat_history):
        """
        Sets the history of chat messages from the current request context.

        Args:
            chat_history (List[ChatMessage]): The chat history associated with the current request context.
        """
        _request_context.chat_history = chat_history

    def clear_agent_context(self):
        """
        Clears the current request context.
        """
        if hasattr(_request_context):
            _request_context.clear()

    def streaming_evaluate_prompt(self, prompt: str, system_message: str = None, llm_name: str = None, max_tokens: int = None):
        """
        Generate response to the prompt using the specified model.
        This works similar to evaluate_prompt, but would stream the result as well so that user is aware of the current status of the generation.

        Args:
            prompt (str): Prompt to use for generation.
            system_message (str): System message for models that support it.
            llm_name (str): Name of the underlying LLM to be used for generation. Should be one of 'gpt-4' or 'gpt-3.5-turbo'. Default is auto selection.
            max_tokens (int): Maximum number of tokens to generate. If set, the model will just stop generating after this token limit is reached.

        Returns:
            LLMResponse: The response from the model, raw text and parsed components.
        """
        llm_parameters = self.get_llm_parameters(
            prompt, system_message=system_message, llm_name=llm_name, max_tokens=max_tokens)
        result = self._stream_llm_call(llm_parameters.parameters)
        result = self._build_class(LlmResponse, result)
        return result

    def _get_agent_async_app_request_id(self):
        """
        Gets the current request ID for the current request context of async app. Applicable within a AIAgent execute function.

        Returns:
            str: The request ID for the current request being processed by the Agent.
        """
        return get_object_from_context(self, _request_context, 'request_id', str)

    def _get_agent_async_app_caller(self):
        """
        Gets the caller for the current request context of async app. Applicable within a AIAgent execute function.

        Returns:
            str: The caller for the current request being processed by the Agent.
        """
        return get_object_from_context(self, _request_context, 'async_app_caller', str)

    def stream_message(self, message: str) -> None:
        """
        Streams a message to the current request context. Applicable within a AIAgent execute function.
        If the request is from the abacus.ai app, the response will be streamed to the UI. otherwise would be logged info if used from notebook or python script.

        Args:
            message (str): The message to be streamed.
        """
        request_id = self._get_agent_async_app_request_id()
        caller = self._get_agent_async_app_caller()
        if not request_id or not caller:
            logging.info(message)
            return
        self._call_aiagent_asyncapp_sync_message(
            request_id, caller, message=message)

    def _stream_llm_call(self, llm_args: dict):
        request_id = self._get_agent_async_app_request_id()
        caller = self._get_agent_async_app_caller()
        if not request_id or not caller:
            raise Exception(
                'Unable to stream LLM call to UI. No request ID or caller found.')
        return self._call_aiagent_asyncapp_sync_message(request_id, caller, llm_args=llm_args)

    def _call_aiagent_asyncapp_sync_message(self, request_id, caller, message=None, llm_args=None):
        """
        Calls the AI Agent AsyncApp sync message endpoint.

        Args:
            request_id (str): The request ID for the current request being processed by the Agent.
            caller (str): The caller for the current request being processed by the Agent.
            message (str): The message to send to the AsyncApp.
            llm_args (dict): The LLM arguments to send to the AsyncApp.

        Returns:
            str: The response from the AsyncApp.
        """
        if not caller.endswith('/'):
            caller = caller + '/'
        api_endpont = f'{caller}sendSyncMessageExecuteAgentRequest'
        body = {'requestId': request_id}
        if message:
            body['message'] = message
        elif llm_args:
            body['llmArgs'] = llm_args
        else:
            raise Exception('Either message or llm_args must be provided.')
        headers = {'APIKEY': self.api_key}
        response = self._request(
            api_endpont, method='POST', body=body, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f'Error calling AI Agent AsyncApp sync message endpoint. Status code: {response.status_code}. Response: {response.text}')

    def add_user_to_organization(self, email: str):
        """Invite a user to your organization. This method will send the specified email address an invitation link to join your organization.

        Args:
            email (str): The email address to invite to your organization."""
        return self._call_api('addUserToOrganization', 'POST', query_params={}, body={'email': email})

    def create_organization_group(self, group_name: str, permissions: list, default_group: bool = False) -> OrganizationGroup:
        """Creates a new Organization Group.

        Args:
            group_name (str): The name of the group.
            permissions (list): The list of permissions to initialize the group with.
            default_group (bool): If True, this group will replace the current default group.

        Returns:
            OrganizationGroup: Information about the created Organization Group."""
        return self._call_api('createOrganizationGroup', 'POST', query_params={}, body={'groupName': group_name, 'permissions': permissions, 'defaultGroup': default_group}, parse_type=OrganizationGroup)

    def add_organization_group_permission(self, organization_group_id: str, permission: str):
        """Adds a permission to the specified Organization Group.

        Args:
            organization_group_id (str): Unique string identifier of the Organization Group.
            permission (str): Permission to add to the Organization Group."""
        return self._call_api('addOrganizationGroupPermission', 'POST', query_params={}, body={'organizationGroupId': organization_group_id, 'permission': permission})

    def remove_organization_group_permission(self, organization_group_id: str, permission: str):
        """Removes a permission from the specified Organization Group.

        Args:
            organization_group_id (str): Unique string identifier of the Organization Group.
            permission (str): The permission to remove from the Organization Group."""
        return self._call_api('removeOrganizationGroupPermission', 'POST', query_params={}, body={'organizationGroupId': organization_group_id, 'permission': permission})

    def delete_organization_group(self, organization_group_id: str):
        """Deletes the specified Organization Group

        Args:
            organization_group_id (str): Unique string identifier of the organization group."""
        return self._call_api('deleteOrganizationGroup', 'DELETE', query_params={'organizationGroupId': organization_group_id})

    def add_user_to_organization_group(self, organization_group_id: str, email: str):
        """Adds a user to the specified Organization Group.

        Args:
            organization_group_id (str): Unique string identifier of the Organization Group.
            email (str): Email of the user to be added to the group."""
        return self._call_api('addUserToOrganizationGroup', 'POST', query_params={}, body={'organizationGroupId': organization_group_id, 'email': email})

    def remove_user_from_organization_group(self, organization_group_id: str, email: str):
        """Removes a user from an Organization Group.

        Args:
            organization_group_id (str): Unique string identifier of the Organization Group.
            email (str): Email of the user to remove."""
        return self._call_api('removeUserFromOrganizationGroup', 'DELETE', query_params={'organizationGroupId': organization_group_id, 'email': email})

    def set_default_organization_group(self, organization_group_id: str):
        """Sets the default Organization Group to which all new users joining an organization are automatically added.

        Args:
            organization_group_id (str): Unique string identifier of the Organization Group."""
        return self._call_api('setDefaultOrganizationGroup', 'POST', query_params={}, body={'organizationGroupId': organization_group_id})

    def delete_api_key(self, api_key_id: str):
        """Delete a specified API key.

        Args:
            api_key_id (str): The ID of the API key to delete."""
        return self._call_api('deleteApiKey', 'DELETE', query_params={'apiKeyId': api_key_id})

    def remove_user_from_organization(self, email: str):
        """Removes the specified user from the organization. You can remove yourself; otherwise, you must be an organization administrator to use this method to remove other users from the organization.

        Args:
            email (str): The email address of the user to remove from the organization."""
        return self._call_api('removeUserFromOrganization', 'DELETE', query_params={'email': email})

    def create_deployment_webhook(self, deployment_id: str, endpoint: str, webhook_event_type: str, payload_template: dict = None) -> Webhook:
        """Create a webhook attached to a given deployment ID.

        Args:
            deployment_id (str): Unique string identifier for the deployment this webhook will attach to.
            endpoint (str): URI that the webhook will send HTTP POST requests to.
            webhook_event_type (str): One of 'DEPLOYMENT_START', 'DEPLOYMENT_SUCCESS', or 'DEPLOYMENT_FAILED'.
            payload_template (dict): Template for the body of the HTTP POST requests. Defaults to {}.

        Returns:
            Webhook: The webhook attached to the deployment."""
        return self._call_api('createDeploymentWebhook', 'POST', query_params={'deploymentId': deployment_id}, body={'endpoint': endpoint, 'webhookEventType': webhook_event_type, 'payloadTemplate': payload_template}, parse_type=Webhook)

    def update_webhook(self, webhook_id: str, endpoint: str = None, webhook_event_type: str = None, payload_template: dict = None):
        """Update the webhook

        Args:
            webhook_id (str): The ID of the webhook to be updated.
            endpoint (str): If provided, changes the webhook's endpoint.
            webhook_event_type (str): If provided, changes the event type.
            payload_template (dict): If provided, changes the payload template."""
        return self._call_api('updateWebhook', 'PATCH', query_params={}, body={'webhookId': webhook_id, 'endpoint': endpoint, 'webhookEventType': webhook_event_type, 'payloadTemplate': payload_template})

    def delete_webhook(self, webhook_id: str):
        """Delete the webhook

        Args:
            webhook_id (str): Unique identifier of the target webhook."""
        return self._call_api('deleteWebhook', 'DELETE', query_params={'webhookId': webhook_id})

    def create_project(self, name: str, use_case: str) -> Project:
        """Creates a project with the specified project name and use case. Creating a project creates a container for all datasets and models associated with a particular problem/project. For example, if you want to create a model to detect fraud, you need to first create a project, upload datasets, create feature groups, and then create one or more models to get predictions for your use case.

        Args:
            name (str): The project's name.
            use_case (str): The use case that the project solves. Refer to our [guide on use cases](https://api.abacus.ai/app/help/useCases) for further details of each use case. The following enums are currently available for you to choose from:  LANGUAGE_DETECTION,  NLP_SENTIMENT,  NLP_SEARCH,  NLP_CHAT,  CHAT_LLM,  NLP_SENTENCE_BOUNDARY_DETECTION,  NLP_CLASSIFICATION,  NLP_SUMMARIZATION,  NLP_DOCUMENT_VISUALIZATION,  AI_AGENT,  EMBEDDINGS_ONLY,  MODEL_WITH_EMBEDDINGS,  TORCH_MODEL,  TORCH_MODEL_WITH_EMBEDDINGS,  PYTHON_MODEL,  NOTEBOOK_PYTHON_MODEL,  DOCKER_MODEL,  DOCKER_MODEL_WITH_EMBEDDINGS,  CUSTOMER_CHURN,  ENERGY,  FINANCIAL_METRICS,  CUMULATIVE_FORECASTING,  FRAUD_ACCOUNT,  FRAUD_THREAT,  FRAUD_TRANSACTIONS,  OPERATIONS_CLOUD,  CLOUD_SPEND,  TIMESERIES_ANOMALY_DETECTION,  OPERATIONS_MAINTENANCE,  OPERATIONS_INCIDENT,  PERS_PROMOTIONS,  PREDICTING,  FEATURE_STORE,  RETAIL,  SALES_FORECASTING,  SALES_SCORING,  FEED_RECOMMEND,  USER_RANKINGS,  NAMED_ENTITY_RECOGNITION,  USER_RECOMMENDATIONS,  USER_RELATED,  VISION,  VISION_REGRESSION,  VISION_OBJECT_DETECTION,  FEATURE_DRIFT,  SCHEDULING,  GENERIC_FORECASTING,  PRETRAINED_IMAGE_TEXT_DESCRIPTION,  PRETRAINED_SPEECH_RECOGNITION,  PRETRAINED_STYLE_TRANSFER,  PRETRAINED_TEXT_TO_IMAGE_GENERATION,  THEME_ANALYSIS,  CLUSTERING,  CLUSTERING_TIMESERIES,  PRETRAINED_INSTRUCT_PIX2PIX,  PRETRAINED_TEXT_CLASSIFICATION.

        Returns:
            Project: This object represents the newly created project."""
        return self._call_api('createProject', 'POST', query_params={}, body={'name': name, 'useCase': use_case}, parse_type=Project)

    def rename_project(self, project_id: str, name: str):
        """This method renames a project after it is created.

        Args:
            project_id (str): The unique identifier for the project.
            name (str): The new name for the project."""
        return self._call_api('renameProject', 'PATCH', query_params={}, body={'projectId': project_id, 'name': name})

    def delete_project(self, project_id: str):
        """Delete a specified project from your organization.

        This method deletes the project, its associated trained models, and deployments. The datasets attached to the specified project remain available for use with other projects in the organization.

        This method will not delete a project that contains active deployments. Ensure that all active deployments are stopped before using the delete option.

        Note: All projects, models, and deployments cannot be recovered once they are deleted.


        Args:
            project_id (str): The unique ID of the project to delete."""
        return self._call_api('deleteProject', 'DELETE', query_params={'projectId': project_id})

    def add_project_tags(self, project_id: str, tags: list):
        """This method adds a tag to a project.

        Args:
            project_id (str): The unique identifier for the project.
            tags (list): The tags to add to the project."""
        return self._call_api('addProjectTags', 'POST', query_params={}, body={'projectId': project_id, 'tags': tags})

    def remove_project_tags(self, project_id: str, tags: list):
        """This method removes a tag from a project.

        Args:
            project_id (str): The unique identifier for the project.
            tags (list): The tags to remove from the project."""
        return self._call_api('removeProjectTags', 'DELETE', query_params={'projectId': project_id, 'tags': tags})

    def add_feature_group_to_project(self, feature_group_id: str, project_id: str, feature_group_type: str = 'CUSTOM_TABLE'):
        """Adds a feature group to a project.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            project_id (str): The unique ID associated with the project.
            feature_group_type (str): The feature group type of the feature group, based on the use case under which the feature group is being created."""
        return self._call_api('addFeatureGroupToProject', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'projectId': project_id, 'featureGroupType': feature_group_type})

    def set_project_feature_group_config(self, feature_group_id: str, project_id: str, project_config: dict = None):
        """Sets a feature group's project config

        Args:
            feature_group_id (str): Unique string identifier for the feature group.
            project_id (str): Unique string identifier for the project.
            project_config (dict): JSON object for the feature group's project configuration."""
        return self._call_api('setProjectFeatureGroupConfig', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'projectId': project_id, 'projectConfig': project_config})

    def remove_feature_group_from_project(self, feature_group_id: str, project_id: str):
        """Removes a feature group from a project.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            project_id (str): The unique ID associated with the project."""
        return self._call_api('removeFeatureGroupFromProject', 'DELETE', query_params={'featureGroupId': feature_group_id, 'projectId': project_id})

    def set_feature_group_type(self, feature_group_id: str, project_id: str, feature_group_type: str = 'CUSTOM_TABLE'):
        """Update the feature group type in a project. The feature group must already be added to the project.

        Args:
            feature_group_id (str): Unique identifier associated with the feature group.
            project_id (str): Unique identifier associated with the project.
            feature_group_type (str): The feature group type to set the feature group as."""
        return self._call_api('setFeatureGroupType', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'projectId': project_id, 'featureGroupType': feature_group_type})

    def set_feature_mapping(self, project_id: str, feature_group_id: str, feature_name: str, feature_mapping: str = None, nested_column_name: str = None) -> List[Feature]:
        """Set a column's feature mapping. If the column mapping is single-use and already set in another column in this feature group, this call will first remove the other column's mapping and move it to this column.

        Args:
            project_id (str): The unique ID associated with the project.
            feature_group_id (str): The unique ID associated with the feature group.
            feature_name (str): The name of the feature.
            feature_mapping (str): The mapping of the feature in the feature group.
            nested_column_name (str): The name of the nested column if the input feature is part of a nested feature group for the given feature_group_id.

        Returns:
            list[Feature]: A list of objects that describes the resulting feature group's schema after the feature's featureMapping is set."""
        return self._call_api('setFeatureMapping', 'POST', query_params={}, body={'projectId': project_id, 'featureGroupId': feature_group_id, 'featureName': feature_name, 'featureMapping': feature_mapping, 'nestedColumnName': nested_column_name}, parse_type=Feature)

    def add_annotation(self, annotation: dict, feature_group_id: str, feature_name: str, doc_id: str = None, feature_group_row_identifier: str = None, annotation_source: str = 'ui', document: str = None, status: str = None, comments: dict = None, project_id: str = None, save_metadata: bool = False) -> AnnotationEntry:
        """Add an annotation entry to the database.

        Args:
            annotation (dict): The annotation to add. Format of the annotation is determined by its annotation type.
            feature_group_id (str): The ID of the feature group the annotation is on.
            feature_name (str): The name of the feature the annotation is on.
            doc_id (str): The ID of the primary document the annotation is on. At least one of the doc_id or feature_group_row_identifier must be provided in order to identify the correct annotation.
            feature_group_row_identifier (str): The key value of the feature group row the annotation is on (cast to string). Usually the feature group's primary / identifier key value. At least one of the doc_id or feature_group_row_identifier must be provided in order to identify the correct annotation.
            annotation_source (str): Indicator of whether the annotation came from the UI, bulk upload, etc.
            document (str): The document the annotation is on. This is optional.
            status (str): The status of the annotation. Can be one of 'todo', 'in_progress', 'done'. This is optional.
            comments (dict): Comments for the annotation. This is a dictionary of feature name to the corresponding comment. This is optional.
            project_id (str): The ID of the project that the annotation is associated with. This is optional.
            save_metadata (bool): Whether to save the metadata for the annotation. This is optional.

        Returns:
            AnnotationEntry: The annotation entry that was added."""
        return self._call_api('addAnnotation', 'POST', query_params={}, body={'annotation': annotation, 'featureGroupId': feature_group_id, 'featureName': feature_name, 'docId': doc_id, 'featureGroupRowIdentifier': feature_group_row_identifier, 'annotationSource': annotation_source, 'document': document, 'status': status, 'comments': comments, 'projectId': project_id, 'saveMetadata': save_metadata}, parse_type=AnnotationEntry)

    def describe_annotation(self, feature_group_id: str, feature_name: str = None, doc_id: str = None, feature_group_row_identifier: str = None) -> AnnotationEntry:
        """Get the latest annotation entry for a given feature group, feature, and document.

        Args:
            feature_group_id (str): The ID of the feature group the annotation is on.
            feature_name (str): The name of the feature the annotation is on.
            doc_id (str): The ID of the primary document the annotation is on. At least one of the doc_id or feature_group_row_identifier must be provided in order to identify the correct annotation.
            feature_group_row_identifier (str): The key value of the feature group row the annotation is on (cast to string). Usually the feature group's primary / identifier key value. At least one of the doc_id or feature_group_row_identifier must be provided in order to identify the correct annotation.

        Returns:
            AnnotationEntry: The latest annotation entry for the given feature group, feature, document, and/or annotation key value."""
        return self._call_api('describeAnnotation', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'featureName': feature_name, 'docId': doc_id, 'featureGroupRowIdentifier': feature_group_row_identifier}, parse_type=AnnotationEntry)

    def update_annotation_status(self, feature_group_id: str, feature_name: str, status: str, doc_id: str = None, feature_group_row_identifier: str = None, save_metadata: bool = False) -> AnnotationEntry:
        """Update the status of an annotation entry.

        Args:
            feature_group_id (str): The ID of the feature group the annotation is on.
            feature_name (str): The name of the feature the annotation is on.
            status (str): The new status of the annotation. Must be one of the following: 'TODO', 'IN_PROGRESS', 'DONE'.
            doc_id (str): The ID of the primary document the annotation is on. At least one of the doc_id or feature_group_row_identifier must be provided in order to identify the correct annotation.
            feature_group_row_identifier (str): The key value of the feature group row the annotation is on (cast to string). Usually the feature group's primary / identifier key value. At least one of the doc_id or feature_group_row_identifier must be provided in order to identify the correct annotation.
            save_metadata (bool): If True, save the metadata for the annotation entry.

        Returns:
            AnnotationEntry: The updated annotation entry."""
        return self._call_api('updateAnnotationStatus', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'featureName': feature_name, 'status': status, 'docId': doc_id, 'featureGroupRowIdentifier': feature_group_row_identifier, 'saveMetadata': save_metadata}, parse_type=AnnotationEntry)

    def get_document_to_annotate(self, feature_group_id: str, feature_name: str, feature_group_row_identifier: str = None, get_previous: bool = False) -> AnnotationEntry:
        """Get an available document that needs to be annotated for a given feature group and feature.

        Args:
            feature_group_id (str): The ID of the feature group the annotation is on.
            feature_name (str): The name of the feature the annotation is on.
            feature_group_row_identifier (str): The key value of the feature group row the annotation is on (cast to string). Usually the primary key value. If provided, fetch the immediate next (or previous) available document.
            get_previous (bool): If True, get the previous document instead of the next document. Applicable if feature_group_row_identifier is provided.

        Returns:
            AnnotationEntry: The latest annotation entry for the given feature group, feature, document, and/or annotation key value."""
        return self._call_api('getDocumentToAnnotate', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'featureName': feature_name, 'featureGroupRowIdentifier': feature_group_row_identifier, 'getPrevious': get_previous}, parse_type=AnnotationEntry)

    def import_annotation_labels(self, feature_group_id: str, file: io.TextIOBase, annotation_type: str) -> AnnotationConfig:
        """Imports annotation labels from csv file. All valid values in the file will be imported as labels (including header row if present).

        Args:
            feature_group_id (str): The unique string identifier of the feature group.
            file (io.TextIOBase): The file to import. Must be a csv file.
            annotation_type (str): The type of the annotation.

        Returns:
            AnnotationConfig: The annotation config for the feature group."""
        return self._call_api('importAnnotationLabels', 'POST', query_params={'featureGroupId': feature_group_id, 'annotationType': annotation_type}, parse_type=AnnotationConfig, files={'file': file})

    def create_feature_group(self, table_name: str, sql: str, description: str = None) -> FeatureGroup:
        """Creates a new FeatureGroup from a SQL statement.

        Args:
            table_name (str): The unique name to be given to the FeatureGroup.
            sql (str): Input SQL statement for forming the FeatureGroup.
            description (str): The description about the FeatureGroup.

        Returns:
            FeatureGroup: The created FeatureGroup."""
        return self._call_api('createFeatureGroup', 'POST', query_params={}, body={'tableName': table_name, 'sql': sql, 'description': description}, parse_type=FeatureGroup)

    def create_feature_group_from_template(self, table_name: str, feature_group_template_id: str, template_bindings: list = None, should_attach_feature_group_to_template: bool = True, description: str = None) -> FeatureGroup:
        """Creates a new feature group from a SQL statement.

        Args:
            table_name (str): The unique name to be given to the feature group.
            feature_group_template_id (str): The unique ID associated with the template that will be used to create this feature group.
            template_bindings (list): Variable bindings that override the template's variable values.
            should_attach_feature_group_to_template (bool): Set to `False` to create a feature group but not leave it attached to the template that created it.
            description (str): A user-friendly description of this feature group.

        Returns:
            FeatureGroup: The created feature group."""
        return self._call_api('createFeatureGroupFromTemplate', 'POST', query_params={}, body={'tableName': table_name, 'featureGroupTemplateId': feature_group_template_id, 'templateBindings': template_bindings, 'shouldAttachFeatureGroupToTemplate': should_attach_feature_group_to_template, 'description': description}, parse_type=FeatureGroup)

    def create_feature_group_from_function(self, table_name: str, function_source_code: str = None, function_name: str = None, input_feature_groups: list = None, description: str = None, cpu_size: str = None, memory: int = None, package_requirements: list = None, use_original_csv_names: bool = False, python_function_name: str = None, python_function_bindings: list = None) -> FeatureGroup:
        """Creates a new feature in a Feature Group from user-provided code. Currently supported code languages are Python.

        If a list of input feature groups are supplied, we will provide DataFrames (pandas, in the case of Python) with the materialized feature groups for those input feature groups as arguments to the function.

        This method expects the source code to be a valid language source file containing a function. This function needs to return a DataFrame when executed; this DataFrame will be used as the materialized version of this feature group table.


        Args:
            table_name (str): The unique name to be given to the feature group.
            function_source_code (str): Contents of a valid source code file in a supported Feature Group specification language (currently only Python). The source code should contain a function called function_name. A list of allowed import and system libraries for each language is specified in the user functions documentation section.
            function_name (str): Name of the function found in the source code that will be executed (on the optional inputs) to materialize this feature group.
            input_feature_groups (list): List of feature groups that are supplied to the function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            description (str): The description for this feature group.
            cpu_size (str): Size of the CPU for the feature group function.
            memory (int): Memory (in GB) for the feature group function.
            package_requirements (list): List of package requirements for the feature group function. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            use_original_csv_names (bool): Defaults to False, if set it uses the original column names for input feature groups from CSV datasets.
            python_function_name (str): Name of Python Function that contains the source code and function arguments.
            python_function_bindings (list): List of arguments to be supplied to the function as parameters in the format [{'name': 'function_argument', 'variable_type': 'FEATURE_GROUP', 'value': 'name_of_feature_group'}].

        Returns:
            FeatureGroup: The created feature group"""
        return self._call_api('createFeatureGroupFromFunction', 'POST', query_params={}, body={'tableName': table_name, 'functionSourceCode': function_source_code, 'functionName': function_name, 'inputFeatureGroups': input_feature_groups, 'description': description, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements, 'useOriginalCsvNames': use_original_csv_names, 'pythonFunctionName': python_function_name, 'pythonFunctionBindings': python_function_bindings}, parse_type=FeatureGroup)

    def create_sampling_feature_group(self, feature_group_id: str, table_name: str, sampling_config: Union[dict, SamplingConfig], description: str = None) -> FeatureGroup:
        """Creates a new Feature Group defined as a sample of rows from another Feature Group.

        For efficiency, sampling is approximate unless otherwise specified. (e.g. the number of rows may vary slightly from what was requested).


        Args:
            feature_group_id (str): The unique ID associated with the pre-existing Feature Group that will be sampled by this new Feature Group. i.e. the input for sampling.
            table_name (str): The unique name to be given to this sampling Feature Group.
            sampling_config (SamplingConfig): Dictionary defining the sampling method and its parameters.
            description (str): A human-readable description of this Feature Group.

        Returns:
            FeatureGroup: The created Feature Group."""
        return self._call_api('createSamplingFeatureGroup', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'tableName': table_name, 'samplingConfig': sampling_config, 'description': description}, parse_type=FeatureGroup)

    def create_merge_feature_group(self, source_feature_group_id: str, table_name: str, merge_config: Union[dict, MergeConfig], description: str = None) -> FeatureGroup:
        """Creates a new feature group defined as the union of other feature group versions.

        Args:
            source_feature_group_id (str): Unique string identifier corresponding to the dataset feature group that will have its versions merged into this feature group.
            table_name (str): Unique string identifier to be given to this merge feature group.
            merge_config (MergeConfig): JSON object defining the merging method and its parameters.
            description (str): Human-readable description of this feature group.

        Returns:
            FeatureGroup: The created feature group.
Description:
Creates a new feature group defined as the union of other feature group versions."""
        return self._call_api('createMergeFeatureGroup', 'POST', query_params={}, body={'sourceFeatureGroupId': source_feature_group_id, 'tableName': table_name, 'mergeConfig': merge_config, 'description': description}, parse_type=FeatureGroup)

    def create_transform_feature_group(self, source_feature_group_id: str, table_name: str, transform_config: dict, description: str = None) -> FeatureGroup:
        """Creates a new Feature Group defined by a pre-defined transform applied to another Feature Group.

        Args:
            source_feature_group_id (str): Unique string identifier corresponding to the Feature Group to which the transformation will be applied.
            table_name (str): Unique string identifier for the transform Feature Group.
            transform_config (dict): JSON object (aka map) defining the transform and its parameters.
            description (str): Human-readable description of the Feature Group.

        Returns:
            FeatureGroup: The created Feature Group."""
        return self._call_api('createTransformFeatureGroup', 'POST', query_params={}, body={'sourceFeatureGroupId': source_feature_group_id, 'tableName': table_name, 'transformConfig': transform_config, 'description': description}, parse_type=FeatureGroup)

    def create_snapshot_feature_group(self, feature_group_version: str, table_name: str) -> FeatureGroup:
        """Creates a Snapshot Feature Group corresponding to a specific Feature Group version.

        Args:
            feature_group_version (str): Unique string identifier associated with the Feature Group version being snapshotted.
            table_name (str): Name for the newly created Snapshot Feature Group table.

        Returns:
            FeatureGroup: Feature Group corresponding to the newly created Snapshot."""
        return self._call_api('createSnapshotFeatureGroup', 'POST', query_params={}, body={'featureGroupVersion': feature_group_version, 'tableName': table_name}, parse_type=FeatureGroup)

    def set_feature_group_sampling_config(self, feature_group_id: str, sampling_config: Union[dict, SamplingConfig]) -> FeatureGroup:
        """Set a FeatureGroup’s sampling to the config values provided, so that the rows the FeatureGroup returns will be a sample of those it would otherwise have returned.

        Args:
            feature_group_id (str): The unique identifier associated with the FeatureGroup.
            sampling_config (SamplingConfig): A JSON string object specifying the sampling method and parameters specific to that sampling method. An empty sampling_config indicates no sampling.

        Returns:
            FeatureGroup: The updated FeatureGroup."""
        return self._call_api('setFeatureGroupSamplingConfig', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'samplingConfig': sampling_config}, parse_type=FeatureGroup)

    def set_feature_group_merge_config(self, feature_group_id: str, merge_config: Union[dict, MergeConfig]) -> FeatureGroup:
        """Set a MergeFeatureGroup’s merge config to the values provided, so that the feature group only returns a bounded range of an incremental dataset.

        Args:
            feature_group_id (str): Unique identifier associated with the feature group.
            merge_config (MergeConfig): JSON object string specifying the merge rule. An empty merge_config will default to only including the latest dataset version.

        Returns:
            FeatureGroup: The updated FeatureGroup."""
        return self._call_api('setFeatureGroupMergeConfig', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'mergeConfig': merge_config}, parse_type=FeatureGroup)

    def set_feature_group_transform_config(self, feature_group_id: str, transform_config: dict):
        """Set a TransformFeatureGroup’s transform config to the values provided.

        Args:
            feature_group_id (str): A unique string identifier associated with the feature group.
            transform_config (dict): A dictionary object specifying the pre-defined transformation."""
        return self._call_api('setFeatureGroupTransformConfig', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'transformConfig': transform_config})

    def set_feature_group_schema(self, feature_group_id: str, schema: list):
        """Creates a new schema and points the feature group to the new feature group schema ID.

        Args:
            feature_group_id (str): Unique string identifier associated with the feature group.
            schema (list): JSON string containing an array of objects with 'name' and 'dataType' properties."""
        return self._call_api('setFeatureGroupSchema', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'schema': schema})

    def create_feature(self, feature_group_id: str, name: str, select_expression: str) -> FeatureGroup:
        """Creates a new feature in a Feature Group from a SQL select statement.

        Args:
            feature_group_id (str): The unique ID associated with the Feature Group.
            name (str): The name of the feature to add.
            select_expression (str): SQL SELECT expression to create the feature.

        Returns:
            FeatureGroup: A Feature Group object with the newly added feature."""
        return self._call_api('createFeature', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'name': name, 'selectExpression': select_expression}, parse_type=FeatureGroup)

    def add_feature_group_tag(self, feature_group_id: str, tag: str):
        """Adds a tag to the feature group

        Args:
            feature_group_id (str): Unique identifier of the feature group.
            tag (str): The tag to add to the feature group."""
        return self._call_api('addFeatureGroupTag', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'tag': tag})

    def remove_feature_group_tag(self, feature_group_id: str, tag: str):
        """Removes a tag from the specified feature group.

        Args:
            feature_group_id (str): Unique string identifier of the feature group.
            tag (str): The tag to remove from the feature group."""
        return self._call_api('removeFeatureGroupTag', 'DELETE', query_params={'featureGroupId': feature_group_id, 'tag': tag})

    def add_annotatable_feature(self, feature_group_id: str, name: str, annotation_type: str) -> FeatureGroup:
        """Add an annotatable feature in a Feature Group

        Args:
            feature_group_id (str): The unique string identifier for the feature group.
            name (str): The name of the feature to add.
            annotation_type (str): The type of annotation to set.

        Returns:
            FeatureGroup: The feature group after the feature has been set"""
        return self._call_api('addAnnotatableFeature', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'name': name, 'annotationType': annotation_type}, parse_type=FeatureGroup)

    def set_feature_as_annotatable_feature(self, feature_group_id: str, feature_name: str, annotation_type: str, feature_group_row_identifier_feature: str = None, doc_id_feature: str = None) -> FeatureGroup:
        """Sets an existing feature as an annotatable feature (Feature that can be annotated).

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            feature_name (str): The name of the feature to set as annotatable.
            annotation_type (str): The type of annotation label to add.
            feature_group_row_identifier_feature (str): The key value of the feature group row the annotation is on (cast to string) and uniquely identifies the feature group row. At least one of the doc_id or key value must be provided so that the correct annotation can be identified.
            doc_id_feature (str): The name of the document ID feature.

        Returns:
            FeatureGroup: A feature group object with the newly added annotatable feature."""
        return self._call_api('setFeatureAsAnnotatableFeature', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'featureName': feature_name, 'annotationType': annotation_type, 'featureGroupRowIdentifierFeature': feature_group_row_identifier_feature, 'docIdFeature': doc_id_feature}, parse_type=FeatureGroup)

    def set_annotation_status_feature(self, feature_group_id: str, feature_name: str) -> FeatureGroup:
        """Sets a feature as the annotation status feature for a feature group.

        Args:
            feature_group_id (str): The ID of the feature group.
            feature_name (str): The name of the feature to set as the annotation status feature.

        Returns:
            FeatureGroup: The updated feature group."""
        return self._call_api('setAnnotationStatusFeature', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'featureName': feature_name}, parse_type=FeatureGroup)

    def unset_feature_as_annotatable_feature(self, feature_group_id: str, feature_name: str) -> FeatureGroup:
        """Unsets a feature as annotatable

        Args:
            feature_group_id (str): The unique string identifier of the feature group.
            feature_name (str): The name of the feature to unset.

        Returns:
            FeatureGroup: The feature group after unsetting the feature"""
        return self._call_api('unsetFeatureAsAnnotatableFeature', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'featureName': feature_name}, parse_type=FeatureGroup)

    def add_feature_group_annotation_label(self, feature_group_id: str, label_name: str, annotation_type: str, label_definition: str = None) -> FeatureGroup:
        """Adds an annotation label

        Args:
            feature_group_id (str): The unique string identifier of the feature group.
            label_name (str): The name of the label.
            annotation_type (str): The type of the annotation to set.
            label_definition (str): the definition of the label.

        Returns:
            FeatureGroup: The feature group after adding the annotation label"""
        return self._call_api('addFeatureGroupAnnotationLabel', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'labelName': label_name, 'annotationType': annotation_type, 'labelDefinition': label_definition}, parse_type=FeatureGroup)

    def remove_feature_group_annotation_label(self, feature_group_id: str, label_name: str) -> FeatureGroup:
        """Removes an annotation label

        Args:
            feature_group_id (str): The unique string identifier of the feature group.
            label_name (str): The name of the label to remove.

        Returns:
            FeatureGroup: The feature group after adding the annotation label"""
        return self._call_api('removeFeatureGroupAnnotationLabel', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'labelName': label_name}, parse_type=FeatureGroup)

    def add_feature_tag(self, feature_group_id: str, feature: str, tag: str):
        """Adds a tag on a feature

        Args:
            feature_group_id (str): The unique string identifier of the feature group.
            feature (str): The feature to set the tag on.
            tag (str): The tag to set on the feature."""
        return self._call_api('addFeatureTag', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'feature': feature, 'tag': tag})

    def remove_feature_tag(self, feature_group_id: str, feature: str, tag: str):
        """Removes a tag from a feature

        Args:
            feature_group_id (str): The unique string identifier of the feature group.
            feature (str): The feature to remove the tag from.
            tag (str): The tag to remove."""
        return self._call_api('removeFeatureTag', 'DELETE', query_params={'featureGroupId': feature_group_id, 'feature': feature, 'tag': tag})

    def create_nested_feature(self, feature_group_id: str, nested_feature_name: str, table_name: str, using_clause: str, where_clause: str = None, order_clause: str = None) -> FeatureGroup:
        """Creates a new nested feature in a feature group from a SQL statement.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            nested_feature_name (str): The name of the feature.
            table_name (str): The table name of the feature group to nest.
            using_clause (str): The SQL join column or logic to join the nested table with the parent.
            where_clause (str): A SQL WHERE statement to filter the nested rows.
            order_clause (str): A SQL clause to order the nested rows.

        Returns:
            FeatureGroup: A feature group object with the newly added nested feature."""
        return self._call_api('createNestedFeature', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'nestedFeatureName': nested_feature_name, 'tableName': table_name, 'usingClause': using_clause, 'whereClause': where_clause, 'orderClause': order_clause}, parse_type=FeatureGroup)

    def update_nested_feature(self, feature_group_id: str, nested_feature_name: str, table_name: str = None, using_clause: str = None, where_clause: str = None, order_clause: str = None, new_nested_feature_name: str = None) -> FeatureGroup:
        """Updates a previously existing nested feature in a feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            nested_feature_name (str): The name of the feature to be updated.
            table_name (str): The name of the table.
            using_clause (str): The SQL join column or logic to join the nested table with the parent.
            where_clause (str): An SQL WHERE statement to filter the nested rows.
            order_clause (str): An SQL clause to order the nested rows.
            new_nested_feature_name (str): New name for the nested feature.

        Returns:
            FeatureGroup: A feature group object with the updated nested feature."""
        return self._call_api('updateNestedFeature', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'nestedFeatureName': nested_feature_name, 'tableName': table_name, 'usingClause': using_clause, 'whereClause': where_clause, 'orderClause': order_clause, 'newNestedFeatureName': new_nested_feature_name}, parse_type=FeatureGroup)

    def delete_nested_feature(self, feature_group_id: str, nested_feature_name: str) -> FeatureGroup:
        """Delete a nested feature.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            nested_feature_name (str): The name of the feature to be deleted.

        Returns:
            FeatureGroup: A feature group object without the specified nested feature."""
        return self._call_api('deleteNestedFeature', 'DELETE', query_params={'featureGroupId': feature_group_id, 'nestedFeatureName': nested_feature_name}, parse_type=FeatureGroup)

    def create_point_in_time_feature(self, feature_group_id: str, feature_name: str, history_table_name: str, aggregation_keys: list, timestamp_key: str, historical_timestamp_key: str, expression: str, lookback_window_seconds: float = None, lookback_window_lag_seconds: float = 0, lookback_count: int = None, lookback_until_position: int = 0) -> FeatureGroup:
        """Creates a new point in time feature in a feature group using another historical feature group, window spec, and aggregate expression.

        We use the aggregation keys and either the lookbackWindowSeconds or the lookbackCount values to perform the window aggregation for every row in the current feature group.

        If the window is specified in seconds, then all rows in the history table which match the aggregation keys and with historicalTimeFeature greater than or equal to lookbackStartCount and less than the value of the current rows timeFeature are considered. An optional lookbackWindowLagSeconds (+ve or -ve) can be used to offset the current value of the timeFeature. If this value is negative, we will look at the future rows in the history table, so care must be taken to ensure that these rows are available in the online context when we are performing a lookup on this feature group. If the window is specified in counts, then we order the historical table rows aligning by time and consider rows from the window where the rank order is greater than or equal to lookbackCount and includes the row just prior to the current one. The lag is specified in terms of positions using lookbackUntilPosition.


        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            feature_name (str): The name of the feature to create.
            history_table_name (str): The table name of the history table.
            aggregation_keys (list): List of keys to use for joining the historical table and performing the window aggregation.
            timestamp_key (str): Name of feature which contains the timestamp value for the point in time feature.
            historical_timestamp_key (str): Name of feature which contains the historical timestamp.
            expression (str): SQL aggregate expression which can convert a sequence of rows into a scalar value.
            lookback_window_seconds (float): If window is specified in terms of time, number of seconds in the past from the current time for start of the window.
            lookback_window_lag_seconds (float): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window. If it is negative, we are looking at the "future" rows in the history table.
            lookback_count (int): If window is specified in terms of count, the start position of the window (0 is the current row).
            lookback_until_position (int): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window by that many rows. If it is negative, we are looking at those many "future" rows in the history table.

        Returns:
            FeatureGroup: A feature group object with the newly added nested feature."""
        return self._call_api('createPointInTimeFeature', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'featureName': feature_name, 'historyTableName': history_table_name, 'aggregationKeys': aggregation_keys, 'timestampKey': timestamp_key, 'historicalTimestampKey': historical_timestamp_key, 'expression': expression, 'lookbackWindowSeconds': lookback_window_seconds, 'lookbackWindowLagSeconds': lookback_window_lag_seconds, 'lookbackCount': lookback_count, 'lookbackUntilPosition': lookback_until_position}, parse_type=FeatureGroup)

    def update_point_in_time_feature(self, feature_group_id: str, feature_name: str, history_table_name: str = None, aggregation_keys: list = None, timestamp_key: str = None, historical_timestamp_key: str = None, expression: str = None, lookback_window_seconds: float = None, lookback_window_lag_seconds: float = None, lookback_count: int = None, lookback_until_position: int = None, new_feature_name: str = None) -> FeatureGroup:
        """Updates an existing Point-in-Time (PiT) feature in a feature group. See `createPointInTimeFeature` for detailed semantics.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            feature_name (str): The name of the feature.
            history_table_name (str): The table name of the history table. If not specified, we use the current table to do a self join.
            aggregation_keys (list): List of keys to use for joining the historical table and performing the window aggregation.
            timestamp_key (str): Name of the feature which contains the timestamp value for the PiT feature.
            historical_timestamp_key (str): Name of the feature which contains the historical timestamp.
            expression (str): SQL Aggregate expression which can convert a sequence of rows into a scalar value.
            lookback_window_seconds (float): If the window is specified in terms of time, the number of seconds in the past from the current time for the start of the window.
            lookback_window_lag_seconds (float): Optional lag to offset the closest point for the window. If it is positive, we delay the start of the window. If it is negative, we are looking at the "future" rows in the history table.
            lookback_count (int): If the window is specified in terms of count, the start position of the window (0 is the current row).
            lookback_until_position (int): Optional lag to offset the closest point for the window. If it is positive, we delay the start of the window by that many rows. If it is negative, we are looking at those many "future" rows in the history table.
            new_feature_name (str): New name for the PiT feature.

        Returns:
            FeatureGroup: A feature group object with the newly added nested feature."""
        return self._call_api('updatePointInTimeFeature', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'featureName': feature_name, 'historyTableName': history_table_name, 'aggregationKeys': aggregation_keys, 'timestampKey': timestamp_key, 'historicalTimestampKey': historical_timestamp_key, 'expression': expression, 'lookbackWindowSeconds': lookback_window_seconds, 'lookbackWindowLagSeconds': lookback_window_lag_seconds, 'lookbackCount': lookback_count, 'lookbackUntilPosition': lookback_until_position, 'newFeatureName': new_feature_name}, parse_type=FeatureGroup)

    def create_point_in_time_group(self, feature_group_id: str, group_name: str, window_key: str, aggregation_keys: list, history_table_name: str = None, history_window_key: str = None, history_aggregation_keys: list = None, lookback_window: float = None, lookback_window_lag: float = 0, lookback_count: int = None, lookback_until_position: int = 0) -> FeatureGroup:
        """Create a Point-in-Time Group

        Args:
            feature_group_id (str): The unique ID associated with the feature group to add the point in time group to.
            group_name (str): The name of the point in time group.
            window_key (str): Name of feature to use for ordering the rows on the source table.
            aggregation_keys (list): List of keys to perform on the source table for the window aggregation.
            history_table_name (str): The table to use for aggregating, if not provided, the source table will be used.
            history_window_key (str): Name of feature to use for ordering the rows on the history table. If not provided, the windowKey from the source table will be used.
            history_aggregation_keys (list): List of keys to use for join the historical table and performing the window aggregation. If not provided, the aggregationKeys from the source table will be used. Must be the same length and order as the source table's aggregationKeys.
            lookback_window (float): Number of seconds in the past from the current time for the start of the window. If 0, the lookback will include all rows.
            lookback_window_lag (float): Optional lag to offset the closest point for the window. If it is positive, the start of the window is delayed. If it is negative, "future" rows in the history table are used.
            lookback_count (int): If window is specified in terms of count, the start position of the window (0 is the current row).
            lookback_until_position (int): Optional lag to offset the closest point for the window. If it is positive, the start of the window is delayed by that many rows. If it is negative, those many "future" rows in the history table are used.

        Returns:
            FeatureGroup: The feature group after the point in time group has been created."""
        return self._call_api('createPointInTimeGroup', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'groupName': group_name, 'windowKey': window_key, 'aggregationKeys': aggregation_keys, 'historyTableName': history_table_name, 'historyWindowKey': history_window_key, 'historyAggregationKeys': history_aggregation_keys, 'lookbackWindow': lookback_window, 'lookbackWindowLag': lookback_window_lag, 'lookbackCount': lookback_count, 'lookbackUntilPosition': lookback_until_position}, parse_type=FeatureGroup)

    def generate_point_in_time_features(self, feature_group_id: str, group_name: str, columns: list, window_functions: list, prefix: str = None) -> FeatureGroup:
        """Generates and adds PIT features given the selected columns to aggregate over, and the operations to include.

        Args:
            feature_group_id (str): Unique string identifier associated with the feature group.
            group_name (str): Name of the point-in-time group.
            columns (list): List of columns to generate point-in-time features for.
            window_functions (list): List of window functions to operate on.
            prefix (str): Prefix for generated features, defaults to group name

        Returns:
            FeatureGroup: Feature group object with newly added point-in-time features."""
        return self._call_api('generatePointInTimeFeatures', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'groupName': group_name, 'columns': columns, 'windowFunctions': window_functions, 'prefix': prefix}, parse_type=FeatureGroup)

    def update_point_in_time_group(self, feature_group_id: str, group_name: str, window_key: str = None, aggregation_keys: list = None, history_table_name: str = None, history_window_key: str = None, history_aggregation_keys: list = None, lookback_window: float = None, lookback_window_lag: float = None, lookback_count: int = None, lookback_until_position: int = None) -> FeatureGroup:
        """Update Point-in-Time Group

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            group_name (str): The name of the point-in-time group.
            window_key (str): Name of feature which contains the timestamp value for the point-in-time feature.
            aggregation_keys (list): List of keys to use for joining the historical table and performing the window aggregation.
            history_table_name (str): The table to use for aggregating, if not provided, the source table will be used.
            history_window_key (str): Name of feature to use for ordering the rows on the history table. If not provided, the windowKey from the source table will be used.
            history_aggregation_keys (list): List of keys to use for joining the historical table and performing the window aggregation. If not provided, the aggregationKeys from the source table will be used. Must be the same length and order as the source table's aggregationKeys.
            lookback_window (float): Number of seconds in the past from the current time for the start of the window.
            lookback_window_lag (float): Optional lag to offset the closest point for the window. If it is positive, the start of the window is delayed. If it is negative, future rows in the history table are looked at.
            lookback_count (int): If window is specified in terms of count, the start position of the window (0 is the current row).
            lookback_until_position (int): Optional lag to offset the closest point for the window. If it is positive, the start of the window is delayed by that many rows. If it is negative, those many future rows in the history table are looked at.

        Returns:
            FeatureGroup: The feature group after the update has been applied."""
        return self._call_api('updatePointInTimeGroup', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'groupName': group_name, 'windowKey': window_key, 'aggregationKeys': aggregation_keys, 'historyTableName': history_table_name, 'historyWindowKey': history_window_key, 'historyAggregationKeys': history_aggregation_keys, 'lookbackWindow': lookback_window, 'lookbackWindowLag': lookback_window_lag, 'lookbackCount': lookback_count, 'lookbackUntilPosition': lookback_until_position}, parse_type=FeatureGroup)

    def delete_point_in_time_group(self, feature_group_id: str, group_name: str) -> FeatureGroup:
        """Delete point in time group

        Args:
            feature_group_id (str): The unique identifier associated with the feature group.
            group_name (str): The name of the point in time group.

        Returns:
            FeatureGroup: The feature group after the point in time group has been deleted."""
        return self._call_api('deletePointInTimeGroup', 'DELETE', query_params={'featureGroupId': feature_group_id, 'groupName': group_name}, parse_type=FeatureGroup)

    def create_point_in_time_group_feature(self, feature_group_id: str, group_name: str, name: str, expression: str) -> FeatureGroup:
        """Create point in time group feature

        Args:
            feature_group_id (str): A unique string identifier associated with the feature group.
            group_name (str): The name of the point-in-time group.
            name (str): The name of the feature to add to the point-in-time group.
            expression (str): A SQL aggregate expression which can convert a sequence of rows into a scalar value.

        Returns:
            FeatureGroup: The feature group after the update has been applied."""
        return self._call_api('createPointInTimeGroupFeature', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'groupName': group_name, 'name': name, 'expression': expression}, parse_type=FeatureGroup)

    def update_point_in_time_group_feature(self, feature_group_id: str, group_name: str, name: str, expression: str) -> FeatureGroup:
        """Update a feature's SQL expression in a point in time group

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            group_name (str): The name of the point-in-time group.
            name (str): The name of the feature to add to the point-in-time group.
            expression (str): SQL aggregate expression which can convert a sequence of rows into a scalar value.

        Returns:
            FeatureGroup: The feature group after the update has been applied."""
        return self._call_api('updatePointInTimeGroupFeature', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'groupName': group_name, 'name': name, 'expression': expression}, parse_type=FeatureGroup)

    def set_feature_type(self, feature_group_id: str, feature: str, feature_type: str) -> Schema:
        """Set the type of a feature in a feature group. Specify the feature group ID, feature name, and feature type, and the method will return the new column with the changes reflected.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            feature (str): The name of the feature.
            feature_type (str): The machine learning type of the data in the feature. Refer to the [guide on feature types](https://api.abacus.ai/app/help/class/FeatureType) for more information.

        Returns:
            Schema: The feature group after the data_type is applied."""
        return self._call_api('setFeatureType', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'feature': feature, 'featureType': feature_type}, parse_type=Schema)

    def invalidate_streaming_feature_group_data(self, feature_group_id: str, invalid_before_timestamp: int):
        """Invalidates all streaming data with timestamp before invalidBeforeTimestamp

        Args:
            feature_group_id (str): Unique string identifier of the streaming feature group to record data to
            invalid_before_timestamp (int): Unix timestamp; any data with a timestamp before this time will be invalidated"""
        return self._call_api('invalidateStreamingFeatureGroupData', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'invalidBeforeTimestamp': invalid_before_timestamp})

    def concatenate_feature_group_data(self, feature_group_id: str, source_feature_group_id: str, merge_type: str = 'UNION', replace_until_timestamp: int = None, skip_materialize: bool = False):
        """Concatenates data from one Feature Group to another. Feature Groups can be merged if their schemas are compatible, they have the special `updateTimestampKey` column, and (if set) the `primaryKey` column. The second operand in the concatenate operation will be appended to the first operand (merge target).

        Args:
            feature_group_id (str): The destination Feature Group.
            source_feature_group_id (str): The Feature Group to concatenate with the destination Feature Group.
            merge_type (str): `UNION` or `INTERSECTION`.
            replace_until_timestamp (int): The UNIX timestamp to specify the point until which we will replace data from the source Feature Group.
            skip_materialize (bool): If `True`, will not materialize the concatenated Feature Group."""
        return self._call_api('concatenateFeatureGroupData', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'sourceFeatureGroupId': source_feature_group_id, 'mergeType': merge_type, 'replaceUntilTimestamp': replace_until_timestamp, 'skipMaterialize': skip_materialize})

    def remove_concatenation_config(self, feature_group_id: str):
        """Removes the concatenation config on a destination feature group.

        Args:
            feature_group_id (str): Unique identifier of the destination feature group to remove the concatenation configuration from."""
        return self._call_api('removeConcatenationConfig', 'DELETE', query_params={'featureGroupId': feature_group_id})

    def set_feature_group_indexing_config(self, feature_group_id: str, primary_key: str = None, update_timestamp_key: str = None, lookup_keys: list = None):
        """Sets various attributes of the feature group used for deployment lookups and streaming updates.

        Args:
            feature_group_id (str): Unique string identifier for the feature group.
            primary_key (str): Name of the feature which defines the primary key of the feature group.
            update_timestamp_key (str): Name of the feature which defines the update timestamp of the feature group. Used in concatenation and primary key deduplication.
            lookup_keys (list): List of feature names which can be used in the lookup API to restrict the computation to a set of dataset rows. These feature names have to correspond to underlying dataset columns."""
        return self._call_api('setFeatureGroupIndexingConfig', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'primaryKey': primary_key, 'updateTimestampKey': update_timestamp_key, 'lookupKeys': lookup_keys})

    def describe_async_feature_group_operation(self, feature_group_operation_run_id: str) -> ExecuteFeatureGroupOperation:
        """Gets the status of the execution of fg operation

        Args:
            feature_group_operation_run_id (str): The unique ID associated with the execution.

        Returns:
            ExecuteFeatureGroupOperation: A dict that contains the execution status"""
        return self._call_api('describeAsyncFeatureGroupOperation', 'POST', query_params={}, body={'featureGroupOperationRunId': feature_group_operation_run_id}, parse_type=ExecuteFeatureGroupOperation)

    def update_feature_group(self, feature_group_id: str, description: str = None) -> FeatureGroup:
        """Modify an existing Feature Group.

        Args:
            feature_group_id (str): Unique identifier associated with the Feature Group.
            description (str): Description of the Feature Group.

        Returns:
            FeatureGroup: Updated Feature Group object."""
        return self._call_api('updateFeatureGroup', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'description': description}, parse_type=FeatureGroup)

    def detach_feature_group_from_template(self, feature_group_id: str) -> FeatureGroup:
        """Update a feature group to detach it from a template.

        Args:
            feature_group_id (str): Unique string identifier associated with the feature group.

        Returns:
            FeatureGroup: The updated feature group."""
        return self._call_api('detachFeatureGroupFromTemplate', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id}, parse_type=FeatureGroup)

    def update_feature_group_template_bindings(self, feature_group_id: str, template_bindings: list = None) -> FeatureGroup:
        """Update the feature group template bindings for a template feature group.

        Args:
            feature_group_id (str): Unique string identifier associated with the feature group.
            template_bindings (list): Values in these bindings override values set in the template.

        Returns:
            FeatureGroup: Updated feature group."""
        return self._call_api('updateFeatureGroupTemplateBindings', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'templateBindings': template_bindings}, parse_type=FeatureGroup)

    def update_feature_group_python_function_bindings(self, feature_group_id: str, python_function_bindings: list):
        """Updates an existing Feature Group's Python function bindings from a user-provided Python Function. If a list of feature groups are supplied within the Python function bindings, we will provide DataFrames (Pandas in the case of Python) with the materialized feature groups for those input feature groups as arguments to the function.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            python_function_bindings (list): List of arguments to be supplied to the function as parameters in the format [{'name': 'function_argument', 'variable_type': 'FEATURE_GROUP', 'value': 'name_of_feature_group'}]."""
        return self._call_api('updateFeatureGroupPythonFunctionBindings', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'pythonFunctionBindings': python_function_bindings})

    def update_feature_group_python_function(self, feature_group_id: str, python_function_name: str, python_function_bindings: list = []):
        """Updates an existing Feature Group's python function from a user provided Python Function. If a list of feature groups are supplied within the python function

        bindings, we will provide as arguments to the function DataFrame's (pandas in the case of Python) with the materialized
        feature groups for those input feature groups.


        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            python_function_name (str): The name of the python function to be associated with the feature group.
            python_function_bindings (list): List of arguments to be supplied to the function as parameters in the format [{'name': 'function_argument', 'variable_type': 'FEATURE_GROUP', 'value': 'name_of_feature_group'}]."""
        return self._call_api('updateFeatureGroupPythonFunction', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'pythonFunctionName': python_function_name, 'pythonFunctionBindings': python_function_bindings})

    def update_feature_group_sql_definition(self, feature_group_id: str, sql: str) -> FeatureGroup:
        """Updates the SQL statement for a feature group.

        Args:
            feature_group_id (str): The unique identifier associated with the feature group.
            sql (str): The input SQL statement for the feature group.

        Returns:
            FeatureGroup: The updated feature group."""
        return self._call_api('updateFeatureGroupSqlDefinition', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'sql': sql}, parse_type=FeatureGroup)

    def update_dataset_feature_group_feature_expression(self, feature_group_id: str, feature_expression: str) -> FeatureGroup:
        """Updates the SQL feature expression for a Dataset FeatureGroup's custom features

        Args:
            feature_group_id (str): The unique identifier associated with the feature group.
            feature_expression (str): The input SQL statement for the feature group.

        Returns:
            FeatureGroup: The updated feature group."""
        return self._call_api('updateDatasetFeatureGroupFeatureExpression', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'featureExpression': feature_expression}, parse_type=FeatureGroup)

    def update_feature_group_function_definition(self, feature_group_id: str, function_source_code: str = None, function_name: str = None, input_feature_groups: list = None, cpu_size: str = None, memory: int = None, package_requirements: list = None, use_original_csv_names: bool = False, python_function_bindings: list = None) -> FeatureGroup:
        """Updates the function definition for a feature group

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            function_source_code (str): Contents of a valid source code file in a supported Feature Group specification language (currently only Python). The source code should contain a function called `function_name`. A list of allowed import and system libraries for each language is specified in the user functions documentation section.
            function_name (str): Name of the function found in the source code that will be executed (on the optional inputs) to materialize this feature group.
            input_feature_groups (list): List of feature groups that are supplied to the function as parameters. Each of the parameters are materialized DataFrames (same type as the functions return value).
            cpu_size (str): Size of the CPU for the feature group function.
            memory (int): Memory (in GB) for the feature group function.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            use_original_csv_names (bool): If set to `True`, feature group uses the original column names for input feature groups from CSV datasets.
            python_function_bindings (list): List of arguments to be supplied to the function as parameters in the format [{'name': 'function_argument', 'variable_type': 'FEATURE_GROUP', 'value': 'name_of_feature_group'}].

        Returns:
            FeatureGroup: The updated feature group."""
        return self._call_api('updateFeatureGroupFunctionDefinition', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'functionSourceCode': function_source_code, 'functionName': function_name, 'inputFeatureGroups': input_feature_groups, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements, 'useOriginalCsvNames': use_original_csv_names, 'pythonFunctionBindings': python_function_bindings}, parse_type=FeatureGroup)

    def update_feature_group_zip(self, feature_group_id: str, function_name: str, module_name: str, input_feature_groups: list = None, cpu_size: str = None, memory: int = None, package_requirements: list = None) -> Upload:
        """Updates the ZIP file for a feature group created using `createFeatureGroupFromZip`.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            function_name (str): The name of the function found in the source code that will be executed (on the optional inputs) to materialize this feature group.
            module_name (str): The path to the file with the feature group function.
            input_feature_groups (list): A list of feature groups that are supplied to the function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            cpu_size (str): The size of the CPU for the feature group function.
            memory (int): The memory (in GB) for the feature group function.
            package_requirements (list): A list of package requirement strings. For example: `['numpy==1.2.3', 'pandas>=1.4.0']`.

        Returns:
            Upload: The Upload to upload the ZIP file to."""
        return self._call_api('updateFeatureGroupZip', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'functionName': function_name, 'moduleName': module_name, 'inputFeatureGroups': input_feature_groups, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements}, parse_type=Upload)

    def update_feature_group_git(self, feature_group_id: str, application_connector_id: str = None, branch_name: str = None, python_root: str = None, function_name: str = None, module_name: str = None, input_feature_groups: list = None, cpu_size: str = None, memory: int = None, package_requirements: list = None) -> FeatureGroup:
        """Updates a feature group created using `createFeatureGroupFromGit`.

        Args:
            feature_group_id (str): Unique string identifier associated with the feature group.
            application_connector_id (str): Unique string identifier associated with the git application connector.
            branch_name (str): Name of the branch in the git repository to be used for training.
            python_root (str): Path from the top level of the git repository to the directory containing the Python source code. If not provided, the default is the root of the git repository.
            function_name (str): Name of the function found in the source code that will be executed (on the optional inputs) to materialize this feature group.
            module_name (str): Path to the file with the feature group function.
            input_feature_groups (list): List of feature groups that are supplied to the function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            cpu_size (str): Size of the cpu for the feature group function.
            memory (int): Memory (in GB) for the feature group function.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].

        Returns:
            FeatureGroup: The updated FeatureGroup."""
        return self._call_api('updateFeatureGroupGit', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'applicationConnectorId': application_connector_id, 'branchName': branch_name, 'pythonRoot': python_root, 'functionName': function_name, 'moduleName': module_name, 'inputFeatureGroups': input_feature_groups, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements}, parse_type=FeatureGroup)

    def update_feature(self, feature_group_id: str, name: str, select_expression: str = None, new_name: str = None) -> FeatureGroup:
        """Modifies an existing feature in a feature group.

        Args:
            feature_group_id (str): Unique identifier of the feature group.
            name (str): Name of the feature to be updated.
            select_expression (str): SQL statement for modifying the feature.
            new_name (str): New name of the feature.

        Returns:
            FeatureGroup: Updated feature group object."""
        return self._call_api('updateFeature', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'name': name, 'selectExpression': select_expression, 'newName': new_name}, parse_type=FeatureGroup)

    def export_feature_group_version_to_file_connector(self, feature_group_version: str, location: str, export_file_format: str, overwrite: bool = False) -> FeatureGroupExport:
        """Export Feature group to File Connector.

        Args:
            feature_group_version (str): Unique string identifier for the feature group instance to export.
            location (str): Cloud file location to export to.
            export_file_format (str): Enum string specifying the file format to export to.
            overwrite (bool): If true and a file exists at this location, this process will overwrite the file.

        Returns:
            FeatureGroupExport: The FeatureGroupExport instance."""
        return self._call_api('exportFeatureGroupVersionToFileConnector', 'POST', query_params={}, body={'featureGroupVersion': feature_group_version, 'location': location, 'exportFileFormat': export_file_format, 'overwrite': overwrite}, parse_type=FeatureGroupExport)

    def export_feature_group_version_to_database_connector(self, feature_group_version: str, database_connector_id: str, object_name: str, write_mode: str, database_feature_mapping: dict, id_column: str = None, additional_id_columns: list = None) -> FeatureGroupExport:
        """Export Feature group to Database Connector.

        Args:
            feature_group_version (str): Unique string identifier for the Feature Group instance to export.
            database_connector_id (str): Unique string identifier for the Database Connector to export to.
            object_name (str): Name of the database object to write to.
            write_mode (str): Enum string indicating whether to use INSERT or UPSERT.
            database_feature_mapping (dict): Key/value pair JSON object of "database connector column" -> "feature name" pairs.
            id_column (str): Required if write_mode is UPSERT. Indicates which database column should be used as the lookup key.
            additional_id_columns (list): For database connectors which support it, additional ID columns to use as a complex key for upserting.

        Returns:
            FeatureGroupExport: The FeatureGroupExport instance."""
        return self._call_api('exportFeatureGroupVersionToDatabaseConnector', 'POST', query_params={}, body={'featureGroupVersion': feature_group_version, 'databaseConnectorId': database_connector_id, 'objectName': object_name, 'writeMode': write_mode, 'databaseFeatureMapping': database_feature_mapping, 'idColumn': id_column, 'additionalIdColumns': additional_id_columns}, parse_type=FeatureGroupExport)

    def export_feature_group_version_to_console(self, feature_group_version: str, export_file_format: str) -> FeatureGroupExport:
        """Export Feature group to console.

        Args:
            feature_group_version (str): Unique string identifier of the Feature Group instance to export.
            export_file_format (str): File format to export to.

        Returns:
            FeatureGroupExport: The FeatureGroupExport instance."""
        return self._call_api('exportFeatureGroupVersionToConsole', 'POST', query_params={}, body={'featureGroupVersion': feature_group_version, 'exportFileFormat': export_file_format}, parse_type=FeatureGroupExport)

    def set_feature_group_modifier_lock(self, feature_group_id: str, locked: bool = True):
        """Lock a feature group to prevent modification.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            locked (bool): Whether to disable or enable feature group modification (True or False)."""
        return self._call_api('setFeatureGroupModifierLock', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'locked': locked})

    def add_user_to_feature_group_modifiers(self, feature_group_id: str, email: str):
        """Adds a user to a feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            email (str): The email address of the user to be added."""
        return self._call_api('addUserToFeatureGroupModifiers', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'email': email})

    def add_organization_group_to_feature_group_modifiers(self, feature_group_id: str, organization_group_id: str):
        """Add OrganizationGroup to a feature group modifiers list

        Args:
            feature_group_id (str): Unique string identifier of the feature group.
            organization_group_id (str): Unique string identifier of the organization group."""
        return self._call_api('addOrganizationGroupToFeatureGroupModifiers', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'organizationGroupId': organization_group_id})

    def remove_user_from_feature_group_modifiers(self, feature_group_id: str, email: str):
        """Removes a user from a specified feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            email (str): The email address of the user to be removed."""
        return self._call_api('removeUserFromFeatureGroupModifiers', 'DELETE', query_params={'featureGroupId': feature_group_id, 'email': email})

    def remove_organization_group_from_feature_group_modifiers(self, feature_group_id: str, organization_group_id: str):
        """Removes an OrganizationGroup from a feature group modifiers list

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            organization_group_id (str): The unique ID associated with the organization group."""
        return self._call_api('removeOrganizationGroupFromFeatureGroupModifiers', 'DELETE', query_params={'featureGroupId': feature_group_id, 'organizationGroupId': organization_group_id})

    def delete_feature(self, feature_group_id: str, name: str) -> FeatureGroup:
        """Removes a feature from the feature group.

        Args:
            feature_group_id (str): Unique string identifier associated with the feature group.
            name (str): Name of the feature to be deleted.

        Returns:
            FeatureGroup: Updated feature group object."""
        return self._call_api('deleteFeature', 'DELETE', query_params={'featureGroupId': feature_group_id, 'name': name}, parse_type=FeatureGroup)

    def delete_feature_group(self, feature_group_id: str):
        """Deletes a Feature Group.

        Args:
            feature_group_id (str): Unique string identifier for the feature group to be removed."""
        return self._call_api('deleteFeatureGroup', 'DELETE', query_params={'featureGroupId': feature_group_id})

    def create_feature_group_version(self, feature_group_id: str, variable_bindings: dict = None) -> FeatureGroupVersion:
        """Creates a snapshot for a specified feature group.

        Args:
            feature_group_id (str): Unique string identifier associated with the feature group.
            variable_bindings (dict): Dictionary defining variable bindings that override parent feature group values.

        Returns:
            FeatureGroupVersion: A feature group version."""
        return self._call_api('createFeatureGroupVersion', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'variableBindings': variable_bindings}, parse_type=FeatureGroupVersion)

    def create_feature_group_template(self, feature_group_id: str, name: str, template_sql: str, template_variables: list, description: str = None, template_bindings: list = None, should_attach_feature_group_to_template: bool = False) -> FeatureGroupTemplate:
        """Create a feature group template.

        Args:
            feature_group_id (str): Unique identifier of the feature group this template was created from.
            name (str): User-friendly name for this feature group template.
            template_sql (str): The template SQL that will be resolved by applying values from the template variables to generate SQL for a feature group.
            template_variables (list): The template variables for resolving the template.
            description (str): Description of this feature group template.
            template_bindings (list): If the feature group will be attached to the newly created template, set these variable bindings on that feature group.
            should_attach_feature_group_to_template (bool): Set to `True` to convert the feature group to a template feature group and attach it to the newly created template.

        Returns:
            FeatureGroupTemplate: The created feature group template."""
        return self._call_api('createFeatureGroupTemplate', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'name': name, 'templateSql': template_sql, 'templateVariables': template_variables, 'description': description, 'templateBindings': template_bindings, 'shouldAttachFeatureGroupToTemplate': should_attach_feature_group_to_template}, parse_type=FeatureGroupTemplate)

    def delete_feature_group_template(self, feature_group_template_id: str):
        """Delete an existing feature group template.

        Args:
            feature_group_template_id (str): Unique string identifier associated with the feature group template."""
        return self._call_api('deleteFeatureGroupTemplate', 'DELETE', query_params={'featureGroupTemplateId': feature_group_template_id})

    def update_feature_group_template(self, feature_group_template_id: str, template_sql: str = None, template_variables: list = None, description: str = None, name: str = None) -> FeatureGroupTemplate:
        """Update a feature group template.

        Args:
            feature_group_template_id (str): Unique identifier of the feature group template to update.
            template_sql (str): If provided, the new value to use for the template SQL.
            template_variables (list): If provided, the new value to use for the template variables.
            description (str): Description of this feature group template.
            name (str): User-friendly name for this feature group template.

        Returns:
            FeatureGroupTemplate: The updated feature group template."""
        return self._call_api('updateFeatureGroupTemplate', 'POST', query_params={}, body={'featureGroupTemplateId': feature_group_template_id, 'templateSql': template_sql, 'templateVariables': template_variables, 'description': description, 'name': name}, parse_type=FeatureGroupTemplate)

    def preview_feature_group_template_resolution(self, feature_group_template_id: str = None, template_bindings: list = None, template_sql: str = None, template_variables: list = None, should_validate: bool = True) -> ResolvedFeatureGroupTemplate:
        """Resolve template sql using template variables and template bindings.

        Args:
            feature_group_template_id (str): Unique string identifier. If specified, use this template, otherwise assume an empty template.
            template_bindings (list): Values to override the template variable values specified by the template.
            template_sql (str): If specified, use this as the template SQL instead of the feature group template's SQL.
            template_variables (list): Template variables to use. If a template is provided, this overrides the template's template variables.
            should_validate (bool): If true, validates the resolved SQL.

        Returns:
            ResolvedFeatureGroupTemplate: The resolved template"""
        return self._call_api('previewFeatureGroupTemplateResolution', 'POST', query_params={}, body={'featureGroupTemplateId': feature_group_template_id, 'templateBindings': template_bindings, 'templateSql': template_sql, 'templateVariables': template_variables, 'shouldValidate': should_validate}, parse_type=ResolvedFeatureGroupTemplate)

    def cancel_upload(self, upload_id: str):
        """Cancels an upload.

        Args:
            upload_id (str): A unique string identifier for the upload."""
        return self._call_api('cancelUpload', 'DELETE', query_params={'uploadId': upload_id})

    def upload_part(self, upload_id: str, part_number: int, part_data: io.TextIOBase) -> UploadPart:
        """Uploads part of a large dataset file from your bucket to our system. Our system currently supports parts of up to 5GB and full files of up to 5TB. Note that each part must be at least 5MB in size, unless it is the last part in the sequence of parts for the full file.

        Args:
            upload_id (str): A unique identifier for this upload.
            part_number (int): The 1-indexed number denoting the position of the file part in the sequence of parts for the full file.
            part_data (io.TextIOBase): The multipart/form-data for the current part of the full file.

        Returns:
            UploadPart: The object 'UploadPart' which encapsulates the hash and the etag for the part that got uploaded."""
        return self._call_api('uploadPart', 'POST', query_params={'uploadId': upload_id, 'partNumber': part_number}, parse_type=UploadPart, files={'partData': part_data})

    def mark_upload_complete(self, upload_id: str) -> Upload:
        """Marks an upload process as complete.

        Args:
            upload_id (str): A unique string identifier for the upload process.

        Returns:
            Upload: The upload object associated with the process, containing details of the file."""
        return self._call_api('markUploadComplete', 'POST', query_params={}, body={'uploadId': upload_id}, parse_type=Upload)

    def create_dataset_from_file_connector(self, table_name: str, location: str, file_format: str = None, refresh_schedule: str = None, csv_delimiter: str = None, filename_column: str = None, start_prefix: str = None, until_prefix: str = None, location_date_format: str = None, date_format_lookback_days: int = None, incremental: bool = False, is_documentset: bool = False, extract_bounding_boxes: bool = False, merge_file_schemas: bool = False, reference_only_documentset: bool = False, parsing_config: Union[dict, ParsingConfig] = None) -> Dataset:
        """Creates a dataset from a file located in a cloud storage, such as Amazon AWS S3, using the specified dataset name and location.

        Args:
            table_name (str): Organization-unique table name or the name of the feature group table to create using the source table.
            location (str): The URI location format of the dataset source. The URI location format needs to be specified to match the `location_date_format` when `location_date_format` is specified. For example, Location = s3://bucket1/dir1/dir2/event_date=YYYY-MM-DD/* when `location_date_format` is specified. The URI location format needs to include both the `start_prefix` and `until_prefix` when both are specified. For example, Location s3://bucket1/dir1/* includes both s3://bucket1/dir1/dir2/event_date=2021-08-02/* and s3://bucket1/dir1/dir2/event_date=2021-08-08/*
            file_format (str): The file format of the dataset.
            refresh_schedule (str): The Cron time string format that describes a schedule to retrieve the latest version of the imported dataset. The time is specified in UTC.
            csv_delimiter (str): If the file format is CSV, use a specific csv delimiter.
            filename_column (str): Adds a new column to the dataset with the external URI path.
            start_prefix (str): The start prefix (inclusive) for a range based search on a cloud storage location URI.
            until_prefix (str): The end prefix (exclusive) for a range based search on a cloud storage location URI.
            location_date_format (str): The date format in which the data is partitioned in the cloud storage location. For example, if the data is partitioned as s3://bucket1/dir1/dir2/event_date=YYYY-MM-DD/dir4/filename.parquet, then the `location_date_format` is YYYY-MM-DD. This format needs to be consistent across all files within the specified location.
            date_format_lookback_days (int): The number of days to look back from the current day for import locations that are date partitioned. For example, import date 2021-06-04 with `date_format_lookback_days` = 3 will retrieve data for all the dates in the range [2021-06-02, 2021-06-04].
            incremental (bool): Signifies if the dataset is an incremental dataset.
            is_documentset (bool): Signifies if the dataset is docstore dataset. A docstore dataset contains documents like images, PDFs, audio files etc. or is tabular data with links to such files.
            extract_bounding_boxes (bool): Signifies whether to extract bounding boxes out of the documents. Only valid if is_documentset if True.
            merge_file_schemas (bool): Signifies if the merge file schema policy is enabled.
            reference_only_documentset (bool): Signifies if the data reference only policy is enabled.
            parsing_config (ParsingConfig): Custom config for dataset parsing.

        Returns:
            Dataset: The dataset created."""
        return self._call_api('createDatasetFromFileConnector', 'POST', query_params={}, body={'tableName': table_name, 'location': location, 'fileFormat': file_format, 'refreshSchedule': refresh_schedule, 'csvDelimiter': csv_delimiter, 'filenameColumn': filename_column, 'startPrefix': start_prefix, 'untilPrefix': until_prefix, 'locationDateFormat': location_date_format, 'dateFormatLookbackDays': date_format_lookback_days, 'incremental': incremental, 'isDocumentset': is_documentset, 'extractBoundingBoxes': extract_bounding_boxes, 'mergeFileSchemas': merge_file_schemas, 'referenceOnlyDocumentset': reference_only_documentset, 'parsingConfig': parsing_config}, parse_type=Dataset)

    def create_dataset_version_from_file_connector(self, dataset_id: str, location: str = None, file_format: str = None, csv_delimiter: str = None, merge_file_schemas: bool = None, parsing_config: Union[dict, ParsingConfig] = None) -> DatasetVersion:
        """Creates a new version of the specified dataset.

        Args:
            dataset_id (str): Unique string identifier associated with the dataset.
            location (str): External URI to import the dataset from. If not specified, the last location will be used.
            file_format (str): File format to be used. If not specified, the service will try to detect the file format.
            csv_delimiter (str): If the file format is CSV, use a specific CSV delimiter.
            merge_file_schemas (bool): Signifies if the merge file schema policy is enabled.
            parsing_config (ParsingConfig): Custom config for dataset parsing.

        Returns:
            DatasetVersion: The new Dataset Version created."""
        return self._call_api('createDatasetVersionFromFileConnector', 'POST', query_params={'datasetId': dataset_id}, body={'location': location, 'fileFormat': file_format, 'csvDelimiter': csv_delimiter, 'mergeFileSchemas': merge_file_schemas, 'parsingConfig': parsing_config}, parse_type=DatasetVersion)

    def create_dataset_from_database_connector(self, table_name: str, database_connector_id: str, object_name: str = None, columns: str = None, query_arguments: str = None, refresh_schedule: str = None, sql_query: str = None, incremental: bool = False, timestamp_column: str = None) -> Dataset:
        """Creates a dataset from a Database Connector.

        Args:
            table_name (str): Organization-unique table name.
            database_connector_id (str): Unique String Identifier of the Database Connector to import the dataset from.
            object_name (str): If applicable, the name/ID of the object in the service to query.
            columns (str): The columns to query from the external service object.
            query_arguments (str): Additional query arguments to filter the data.
            refresh_schedule (str): The Cron time string format that describes a schedule to retrieve the latest version of the imported dataset. The time is specified in UTC.
            sql_query (str): The full SQL query to use when fetching data. If present, this parameter will override `object_name`, `columns`, `timestamp_column`, and `query_arguments`.
            incremental (bool): Signifies if the dataset is an incremental dataset.
            timestamp_column (str): If dataset is incremental, this is the column name of the required column in the dataset. This column must contain timestamps in descending order which are used to determine the increments of the incremental dataset.

        Returns:
            Dataset: The created dataset."""
        return self._call_api('createDatasetFromDatabaseConnector', 'POST', query_params={}, body={'tableName': table_name, 'databaseConnectorId': database_connector_id, 'objectName': object_name, 'columns': columns, 'queryArguments': query_arguments, 'refreshSchedule': refresh_schedule, 'sqlQuery': sql_query, 'incremental': incremental, 'timestampColumn': timestamp_column}, parse_type=Dataset)

    def create_dataset_from_application_connector(self, table_name: str, application_connector_id: str, object_id: str = None, start_timestamp: int = None, end_timestamp: int = None, refresh_schedule: str = None) -> Dataset:
        """Creates a dataset from an Application Connector.

        Args:
            table_name (str): Organization-unique table name
            application_connector_id (str): Unique string identifier of the application connector to download data from
            object_id (str): If applicable, the ID of the object in the service to query.
            start_timestamp (int): Unix timestamp of the start of the period that will be queried.
            end_timestamp (int): Unix timestamp of the end of the period that will be queried.
            refresh_schedule (str): Cron time string format that describes a schedule to retrieve the latest version of the imported dataset. The time is specified in UTC.

        Returns:
            Dataset: The created dataset."""
        return self._call_api('createDatasetFromApplicationConnector', 'POST', query_params={}, body={'tableName': table_name, 'applicationConnectorId': application_connector_id, 'objectId': object_id, 'startTimestamp': start_timestamp, 'endTimestamp': end_timestamp, 'refreshSchedule': refresh_schedule}, parse_type=Dataset)

    def create_dataset_version_from_database_connector(self, dataset_id: str, object_name: str = None, columns: str = None, query_arguments: str = None, sql_query: str = None) -> DatasetVersion:
        """Creates a new version of the specified dataset.

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            object_name (str): The name/ID of the object in the service to query. If not specified, the last name will be used.
            columns (str): The columns to query from the external service object. If not specified, the last columns will be used.
            query_arguments (str): Additional query arguments to filter the data. If not specified, the last arguments will be used.
            sql_query (str): The full SQL query to use when fetching data. If present, this parameter will override object_name, columns, and query_arguments.

        Returns:
            DatasetVersion: The new Dataset Version created."""
        return self._call_api('createDatasetVersionFromDatabaseConnector', 'POST', query_params={'datasetId': dataset_id}, body={'objectName': object_name, 'columns': columns, 'queryArguments': query_arguments, 'sqlQuery': sql_query}, parse_type=DatasetVersion)

    def create_dataset_version_from_application_connector(self, dataset_id: str, object_id: str = None, start_timestamp: int = None, end_timestamp: int = None) -> DatasetVersion:
        """Creates a new version of the specified dataset.

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            object_id (str): The ID of the object in the service to query. If not specified, the last name will be used.
            start_timestamp (int): The Unix timestamp of the start of the period that will be queried.
            end_timestamp (int): The Unix timestamp of the end of the period that will be queried.

        Returns:
            DatasetVersion: The new Dataset Version created."""
        return self._call_api('createDatasetVersionFromApplicationConnector', 'POST', query_params={'datasetId': dataset_id}, body={'objectId': object_id, 'startTimestamp': start_timestamp, 'endTimestamp': end_timestamp}, parse_type=DatasetVersion)

    def create_dataset_from_upload(self, table_name: str, file_format: str = None, csv_delimiter: str = None, is_documentset: bool = False, extract_bounding_boxes: bool = False, parsing_config: Union[dict, ParsingConfig] = None, merge_file_schemas: bool = False) -> Upload:
        """Creates a dataset and returns an upload ID that can be used to upload a file.

        Args:
            table_name (str): Organization-unique table name for this dataset.
            file_format (str): The file format of the dataset.
            csv_delimiter (str): If the file format is CSV, use a specific CSV delimiter.
            is_documentset (bool): Signifies if the dataset is a docstore dataset. A docstore dataset contains documents like images, PDFs, audio files etc. or is tabular data with links to such files.
            extract_bounding_boxes (bool): Signifies whether to extract bounding boxes out of the documents. Only valid if is_documentset if True.
            parsing_config (ParsingConfig): Custom config for dataset parsing.
            merge_file_schemas (bool): Signifies whether to merge the schemas of all files in the dataset.

        Returns:
            Upload: A reference to be used when uploading file parts."""
        return self._call_api('createDatasetFromUpload', 'POST', query_params={}, body={'tableName': table_name, 'fileFormat': file_format, 'csvDelimiter': csv_delimiter, 'isDocumentset': is_documentset, 'extractBoundingBoxes': extract_bounding_boxes, 'parsingConfig': parsing_config, 'mergeFileSchemas': merge_file_schemas}, parse_type=Upload)

    def create_dataset_version_from_upload(self, dataset_id: str, file_format: str = None) -> Upload:
        """Creates a new version of the specified dataset using a local file upload.

        Args:
            dataset_id (str): Unique string identifier associated with the dataset.
            file_format (str): File format to be used. If not specified, the service will attempt to detect the file format.

        Returns:
            Upload: Token to be used when uploading file parts."""
        return self._call_api('createDatasetVersionFromUpload', 'POST', query_params={'datasetId': dataset_id}, body={'fileFormat': file_format}, parse_type=Upload)

    def create_streaming_dataset(self, table_name: str) -> Dataset:
        """Creates a streaming dataset. Use a streaming dataset if your dataset is receiving information from multiple sources over an extended period of time.

        Args:
            table_name (str): The feature group table name to create for this dataset.

        Returns:
            Dataset: The streaming dataset created."""
        return self._call_api('createStreamingDataset', 'POST', query_params={}, body={'tableName': table_name}, parse_type=Dataset)

    def snapshot_streaming_data(self, dataset_id: str) -> DatasetVersion:
        """Snapshots the current data in the streaming dataset.

        Args:
            dataset_id (str): The unique ID associated with the dataset.

        Returns:
            DatasetVersion: The new Dataset Version created by taking a snapshot of the current data in the streaming dataset."""
        return self._call_api('snapshotStreamingData', 'POST', query_params={'datasetId': dataset_id}, body={}, parse_type=DatasetVersion)

    def set_dataset_column_data_type(self, dataset_id: str, column: str, data_type: str) -> Dataset:
        """Set a Dataset's column type.

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            column (str): The name of the column.
            data_type (str): The type of the data in the column. Refer to the [guide on data types](https://api.abacus.ai/app/help/class/DataType) for more information. Note: Some ColumnMappings may restrict the options or explicitly set the DataType.

        Returns:
            Dataset: The dataset and schema after the data type has been set."""
        return self._call_api('setDatasetColumnDataType', 'POST', query_params={'datasetId': dataset_id}, body={'column': column, 'dataType': data_type}, parse_type=Dataset)

    def create_dataset_from_streaming_connector(self, table_name: str, streaming_connector_id: str, streaming_args: dict = None, refresh_schedule: str = None) -> Dataset:
        """Creates a dataset from a Streaming Connector

        Args:
            table_name (str): Organization-unique table name
            streaming_connector_id (str): Unique String Identifier for the Streaming Connector to import the dataset from
            streaming_args (dict): Dictionary of arguments to read data from the streaming connector
            refresh_schedule (str): Cron time string format that describes a schedule to retrieve the latest version of the imported dataset. Time is specified in UTC.

        Returns:
            Dataset: The created dataset."""
        return self._call_api('createDatasetFromStreamingConnector', 'POST', query_params={}, body={'tableName': table_name, 'streamingConnectorId': streaming_connector_id, 'streamingArgs': streaming_args, 'refreshSchedule': refresh_schedule}, parse_type=Dataset)

    def set_streaming_retention_policy(self, dataset_id: str, retention_hours: int = None, retention_row_count: int = None, ignore_records_before_timestamp: int = None):
        """Sets the streaming retention policy.

        Args:
            dataset_id (str): Unique string identifier for the streaming dataset.
            retention_hours (int): Number of hours to retain streamed data in memory.
            retention_row_count (int): Number of rows to retain streamed data in memory.
            ignore_records_before_timestamp (int): The Unix timestamp (in seconds) to use as a cutoff to ignore all entries sent before it"""
        return self._call_api('setStreamingRetentionPolicy', 'POST', query_params={'datasetId': dataset_id}, body={'retentionHours': retention_hours, 'retentionRowCount': retention_row_count, 'ignoreRecordsBeforeTimestamp': ignore_records_before_timestamp})

    def rename_database_connector(self, database_connector_id: str, name: str):
        """Renames a Database Connector

        Args:
            database_connector_id (str): The unique identifier for the database connector.
            name (str): The new name for the Database Connector."""
        return self._call_api('renameDatabaseConnector', 'PATCH', query_params={}, body={'databaseConnectorId': database_connector_id, 'name': name})

    def rename_application_connector(self, application_connector_id: str, name: str):
        """Renames a Application Connector

        Args:
            application_connector_id (str): The unique identifier for the application connector.
            name (str): A new name for the application connector."""
        return self._call_api('renameApplicationConnector', 'PATCH', query_params={}, body={'applicationConnectorId': application_connector_id, 'name': name})

    def verify_database_connector(self, database_connector_id: str):
        """Checks if Abacus.AI can access the specified database.

        Args:
            database_connector_id (str): Unique string identifier for the database connector."""
        return self._call_api('verifyDatabaseConnector', 'POST', query_params={}, body={'databaseConnectorId': database_connector_id})

    def verify_file_connector(self, bucket: str) -> FileConnectorVerification:
        """Checks to see if Abacus.AI can access the given bucket.

        Args:
            bucket (str): The bucket to test.

        Returns:
            FileConnectorVerification: The result of the verification."""
        return self._call_api('verifyFileConnector', 'POST', query_params={}, body={'bucket': bucket}, parse_type=FileConnectorVerification)

    def delete_database_connector(self, database_connector_id: str):
        """Delete a database connector.

        Args:
            database_connector_id (str): The unique identifier for the database connector."""
        return self._call_api('deleteDatabaseConnector', 'DELETE', query_params={'databaseConnectorId': database_connector_id})

    def delete_application_connector(self, application_connector_id: str):
        """Delete an application connector.

        Args:
            application_connector_id (str): The unique identifier for the application connector."""
        return self._call_api('deleteApplicationConnector', 'DELETE', query_params={'applicationConnectorId': application_connector_id})

    def delete_file_connector(self, bucket: str):
        """Deletes a file connector

        Args:
            bucket (str): The fully qualified URI of the bucket to remove."""
        return self._call_api('deleteFileConnector', 'DELETE', query_params={'bucket': bucket})

    def verify_application_connector(self, application_connector_id: str):
        """Checks if Abacus.AI can access the application using the provided application connector ID.

        Args:
            application_connector_id (str): Unique string identifier for the application connector."""
        return self._call_api('verifyApplicationConnector', 'POST', query_params={}, body={'applicationConnectorId': application_connector_id})

    def set_azure_blob_connection_string(self, bucket: str, connection_string: str) -> FileConnectorVerification:
        """Authenticates the specified Azure Blob Storage bucket using an authenticated Connection String.

        Args:
            bucket (str): The fully qualified Azure Blob Storage Bucket URI.
            connection_string (str): The Connection String {product_name} should use to authenticate when accessing this bucket.

        Returns:
            FileConnectorVerification: An object with the roleArn and verification status for the specified bucket."""
        return self._call_api('setAzureBlobConnectionString', 'POST', query_params={}, body={'bucket': bucket, 'connectionString': connection_string}, parse_type=FileConnectorVerification)

    def verify_streaming_connector(self, streaming_connector_id: str):
        """Checks to see if Abacus.AI can access the streaming connector.

        Args:
            streaming_connector_id (str): Unique string identifier for the streaming connector to be checked for Abacus.AI access."""
        return self._call_api('verifyStreamingConnector', 'POST', query_params={}, body={'streamingConnectorId': streaming_connector_id})

    def rename_streaming_connector(self, streaming_connector_id: str, name: str):
        """Renames a Streaming Connector

        Args:
            streaming_connector_id (str): The unique identifier for the streaming connector.
            name (str): A new name for the streaming connector."""
        return self._call_api('renameStreamingConnector', 'PATCH', query_params={}, body={'streamingConnectorId': streaming_connector_id, 'name': name})

    def delete_streaming_connector(self, streaming_connector_id: str):
        """Delete a streaming connector.

        Args:
            streaming_connector_id (str): The unique identifier for the streaming connector."""
        return self._call_api('deleteStreamingConnector', 'DELETE', query_params={'streamingConnectorId': streaming_connector_id})

    def create_streaming_token(self) -> StreamingAuthToken:
        """Creates a streaming token for the specified project. Streaming tokens are used to authenticate requests when appending data to streaming datasets.

        Returns:
            StreamingAuthToken: The generated streaming token."""
        return self._call_api('createStreamingToken', 'POST', query_params={}, body={}, parse_type=StreamingAuthToken)

    def delete_streaming_token(self, streaming_token: str):
        """Deletes the specified streaming token.

        Args:
            streaming_token (str): The streaming token to delete."""
        return self._call_api('deleteStreamingToken', 'DELETE', query_params={'streamingToken': streaming_token})

    def delete_dataset(self, dataset_id: str):
        """Deletes the specified dataset from the organization.

        Args:
            dataset_id (str): Unique string identifier of the dataset to delete."""
        return self._call_api('deleteDataset', 'DELETE', query_params={'datasetId': dataset_id})

    def get_training_config_options(self, project_id: str, feature_group_ids: list = None, for_retrain: bool = False, current_training_config: Union[dict, TrainingConfig] = None, is_additional_model: bool = False) -> List[TrainingConfigOptions]:
        """Retrieves the full initial description of the model training configuration options available for the specified project. The configuration options available are determined by the use case associated with the specified project. Refer to the [Use Case Documentation]({USE_CASES_URL}) for more information on use cases and use case-specific configuration options.

        Args:
            project_id (str): The unique ID associated with the project.
            feature_group_ids (list): The feature group IDs to be used for training.
            for_retrain (bool): Whether the training config options are used for retraining.
            current_training_config (TrainingConfig): The current state of the training config, with some options set, which shall be used to get new options after refresh. This is `None` by default initially.
            is_additional_model (bool): Whether to get training config options for an additional model

        Returns:
            list[TrainingConfigOptions]: An array of options that can be specified when training a model in this project."""
        return self._call_api('getTrainingConfigOptions', 'POST', query_params={}, body={'projectId': project_id, 'featureGroupIds': feature_group_ids, 'forRetrain': for_retrain, 'currentTrainingConfig': current_training_config, 'isAdditionalModel': is_additional_model}, parse_type=TrainingConfigOptions)

    def create_train_test_data_split_feature_group(self, project_id: str, training_config: Union[dict, TrainingConfig], feature_group_ids: list) -> FeatureGroup:
        """Get the train and test data split without training the model. Only supported for models with custom algorithms.

        Args:
            project_id (str): The unique ID associated with the project.
            training_config (TrainingConfig): The training config used to influence how the split is calculated.
            feature_group_ids (list): List of feature group IDs provided by the user, including the required one for data split and others to influence how to split.

        Returns:
            FeatureGroup: The feature group containing the training data and folds information."""
        return self._call_api('createTrainTestDataSplitFeatureGroup', 'POST', query_params={}, body={'projectId': project_id, 'trainingConfig': training_config, 'featureGroupIds': feature_group_ids}, parse_type=FeatureGroup)

    def train_model(self, project_id: str, name: str = None, training_config: Union[dict, TrainingConfig] = None, feature_group_ids: list = None, refresh_schedule: str = None, custom_algorithms: list = None, custom_algorithms_only: bool = False, custom_algorithm_configs: dict = None, builtin_algorithms: list = None, cpu_size: str = None, memory: int = None, algorithm_training_configs: list = None) -> Model:
        """Train a model for the specified project.

        This method trains a model in the given project, using user-specified training configurations defined in the `getTrainingConfigOptions` method.


        Args:
            project_id (str): The unique ID associated with the project.
            name (str): The name of the model. Defaults to "<Project Name> Model".
            training_config (TrainingConfig): The training config used to train this model.
            feature_group_ids (list): List of feature group IDs provided by the user to train the model on.
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically retrain the created model.
            custom_algorithms (list): List of user-defined algorithms to train. If not set, the default enabled custom algorithms will be used.
            custom_algorithms_only (bool): Whether to only run custom algorithms.
            custom_algorithm_configs (dict): Configs for each user-defined algorithm; key is the algorithm name, value is the config serialized to JSON.
            builtin_algorithms (list): List of IDs of the builtin algorithms provided by Abacus.AI to train. If not set, all applicable builtin algorithms will be used.
            cpu_size (str): Size of the CPU for the user-defined algorithms during training.
            memory (int): Memory (in GB) for the user-defined algorithms during training.
            algorithm_training_configs (list): List of algorithm specifc training configs that will be part of the model training AutoML run.

        Returns:
            Model: The new model which is being trained."""
        return self._call_api('trainModel', 'POST', query_params={}, body={'projectId': project_id, 'name': name, 'trainingConfig': training_config, 'featureGroupIds': feature_group_ids, 'refreshSchedule': refresh_schedule, 'customAlgorithms': custom_algorithms, 'customAlgorithmsOnly': custom_algorithms_only, 'customAlgorithmConfigs': custom_algorithm_configs, 'builtinAlgorithms': builtin_algorithms, 'cpuSize': cpu_size, 'memory': memory, 'algorithmTrainingConfigs': algorithm_training_configs}, parse_type=Model)

    def create_model_from_python(self, project_id: str, function_source_code: str, train_function_name: str, training_input_tables: list, predict_function_name: str = None, predict_many_function_name: str = None, initialize_function_name: str = None, name: str = None, cpu_size: str = None, memory: int = None, training_config: Union[dict, TrainingConfig] = None, exclusive_run: bool = False, package_requirements: list = None, use_gpu: bool = False) -> Model:
        """Initializes a new Model from user-provided Python code. If a list of input feature groups is supplied, they will be provided as arguments to the train and predict functions with the materialized feature groups for those input feature groups.

        This method expects `functionSourceCode` to be a valid language source file which contains the functions named `trainFunctionName` and `predictFunctionName`. `trainFunctionName` returns the ModelVersion that is the result of training the model using `trainFunctionName` and `predictFunctionName` has no well-defined return type, as it returns the prediction made by the `predictFunctionName`, which can be anything.


        Args:
            project_id (str): The unique ID associated with the project.
            function_source_code (str): Contents of a valid Python source code file. The source code should contain the functions named `trainFunctionName` and `predictFunctionName`. A list of allowed import and system libraries for each language is specified in the user functions documentation section.
            train_function_name (str): Name of the function found in the source code that will be executed to train the model. It is not executed when this function is run.
            training_input_tables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            predict_function_name (str): Name of the function found in the source code that will be executed to run predictions through the model. It is not executed when this function is run.
            predict_many_function_name (str): Name of the function found in the source code that will be executed for batch prediction of the model. It is not executed when this function is run.
            initialize_function_name (str): Name of the function found in the source code to initialize the trained model before using it to make predictions using the model
            name (str): The name you want your model to have. Defaults to "<Project Name> Model"
            cpu_size (str): Size of the CPU for the model training function
            memory (int): Memory (in GB) for the model training function
            training_config (TrainingConfig): Training configuration
            exclusive_run (bool): Decides if this model will be run exclusively or along with other Abacus.ai algorithms
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            use_gpu (bool): Whether this model needs gpu

        Returns:
            Model: The new model, which has not been trained."""
        return self._call_api('createModelFromPython', 'POST', query_params={}, body={'projectId': project_id, 'functionSourceCode': function_source_code, 'trainFunctionName': train_function_name, 'trainingInputTables': training_input_tables, 'predictFunctionName': predict_function_name, 'predictManyFunctionName': predict_many_function_name, 'initializeFunctionName': initialize_function_name, 'name': name, 'cpuSize': cpu_size, 'memory': memory, 'trainingConfig': training_config, 'exclusiveRun': exclusive_run, 'packageRequirements': package_requirements, 'useGpu': use_gpu}, parse_type=Model)

    def rename_model(self, model_id: str, name: str):
        """Renames a model

        Args:
            model_id (str): Unique identifier of the model to rename.
            name (str): The new name to assign to the model."""
        return self._call_api('renameModel', 'PATCH', query_params={}, body={'modelId': model_id, 'name': name})

    def update_python_model(self, model_id: str, function_source_code: str = None, train_function_name: str = None, predict_function_name: str = None, predict_many_function_name: str = None, initialize_function_name: str = None, training_input_tables: list = None, cpu_size: str = None, memory: int = None, package_requirements: list = None, use_gpu: bool = None) -> Model:
        """Updates an existing Python Model using user-provided Python code. If a list of input feature groups is supplied, they will be provided as arguments to the `train` and `predict` functions with the materialized feature groups for those input feature groups.

        This method expects `functionSourceCode` to be a valid language source file which contains the functions named `trainFunctionName` and `predictFunctionName`. `trainFunctionName` returns the ModelVersion that is the result of training the model using `trainFunctionName`. `predictFunctionName` has no well-defined return type, as it returns the prediction made by the `predictFunctionName`, which can be anything.


        Args:
            model_id (str): The unique ID associated with the Python model to be changed.
            function_source_code (str): Contents of a valid Python source code file. The source code should contain the functions named `trainFunctionName` and `predictFunctionName`. A list of allowed import and system libraries for each language is specified in the user functions documentation section.
            train_function_name (str): Name of the function found in the source code that will be executed to train the model. It is not executed when this function is run.
            predict_function_name (str): Name of the function found in the source code that will be executed to run predictions through the model. It is not executed when this function is run.
            predict_many_function_name (str): Name of the function found in the source code that will be executed to run batch predictions through the model. It is not executed when this function is run.
            initialize_function_name (str): Name of the function found in the source code to initialize the trained model before using it to make predictions using the model.
            training_input_tables (list): List of feature groups that are supplied to the `train` function as parameters. Each of the parameters are materialized DataFrames (same type as the functions return value).
            cpu_size (str): Size of the CPU for the model training function.
            memory (int): Memory (in GB) for the model training function.
            package_requirements (list): List of package requirement strings. For example: `['numpy==1.2.3', 'pandas>=1.4.0']`.
            use_gpu (bool): Whether this model needs gpu

        Returns:
            Model: The updated model."""
        return self._call_api('updatePythonModel', 'POST', query_params={}, body={'modelId': model_id, 'functionSourceCode': function_source_code, 'trainFunctionName': train_function_name, 'predictFunctionName': predict_function_name, 'predictManyFunctionName': predict_many_function_name, 'initializeFunctionName': initialize_function_name, 'trainingInputTables': training_input_tables, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements, 'useGpu': use_gpu}, parse_type=Model)

    def update_python_model_zip(self, model_id: str, train_function_name: str = None, predict_function_name: str = None, predict_many_function_name: str = None, train_module_name: str = None, predict_module_name: str = None, training_input_tables: list = None, cpu_size: str = None, memory: int = None, package_requirements: list = None, use_gpu: bool = None) -> Upload:
        """Updates an existing Python Model using a provided zip file. If a list of input feature groups are supplied, they will be provided as arguments to the train and predict functions with the materialized feature groups for those input feature groups.

        This method expects `trainModuleName` and `predictModuleName` to be valid language source files which contain the functions named `trainFunctionName` and `predictFunctionName`, respectively. `trainFunctionName` returns the ModelVersion that is the result of training the model using `trainFunctionName`, and `predictFunctionName` has no well-defined return type, as it returns the prediction made by the `predictFunctionName`, which can be anything.


        Args:
            model_id (str): The unique ID associated with the Python model to be changed.
            train_function_name (str): Name of the function found in the train module that will be executed to train the model. It is not executed when this function is run.
            predict_function_name (str): Name of the function found in the predict module that will be executed to run predictions through the model. It is not executed when this function is run.
            predict_many_function_name (str): Name of the function found in the predict module that will be executed to run batch predictions through the model. It is not executed when this function is run.
            train_module_name (str): Full path of the module that contains the train function from the root of the zip.
            predict_module_name (str): Full path of the module that contains the predict function from the root of the zip.
            training_input_tables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the function's return value).
            cpu_size (str): Size of the CPU for the model training function.
            memory (int): Memory (in GB) for the model training function.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            use_gpu (bool): Whether this model needs gpu

        Returns:
            Upload: The updated model."""
        return self._call_api('updatePythonModelZip', 'POST', query_params={}, body={'modelId': model_id, 'trainFunctionName': train_function_name, 'predictFunctionName': predict_function_name, 'predictManyFunctionName': predict_many_function_name, 'trainModuleName': train_module_name, 'predictModuleName': predict_module_name, 'trainingInputTables': training_input_tables, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements, 'useGpu': use_gpu}, parse_type=Upload)

    def update_python_model_git(self, model_id: str, application_connector_id: str = None, branch_name: str = None, python_root: str = None, train_function_name: str = None, predict_function_name: str = None, predict_many_function_name: str = None, train_module_name: str = None, predict_module_name: str = None, training_input_tables: list = None, cpu_size: str = None, memory: int = None, use_gpu: bool = None) -> Model:
        """Updates an existing Python model using an existing Git application connector. If a list of input feature groups are supplied, these will be provided as arguments to the train and predict functions with the materialized feature groups for those input feature groups.

        This method expects `trainModuleName` and `predictModuleName` to be valid language source files which contain the functions named `trainFunctionName` and `predictFunctionName`, respectively. `trainFunctionName` returns the `ModelVersion` that is the result of training the model using `trainFunctionName`, and `predictFunctionName` has no well-defined return type, as it returns the prediction made by the `predictFunctionName`, which can be anything.


        Args:
            model_id (str): The unique ID associated with the Python model to be changed.
            application_connector_id (str): The unique ID associated with the Git application connector.
            branch_name (str): Name of the branch in the Git repository to be used for training.
            python_root (str): Path from the top level of the Git repository to the directory containing the Python source code. If not provided, the default is the root of the Git repository.
            train_function_name (str): Name of the function found in train module that will be executed to train the model. It is not executed when this function is run.
            predict_function_name (str): Name of the function found in the predict module that will be executed to run predictions through model. It is not executed when this function is run.
            predict_many_function_name (str): Name of the function found in the predict module that will be executed to run batch predictions through model. It is not executed when this function is run.
            train_module_name (str): Full path of the module that contains the train function from the root of the zip.
            predict_module_name (str): Full path of the module that contains the predict function from the root of the zip.
            training_input_tables (list): List of feature groups that are supplied to the train function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            cpu_size (str): Size of the CPU for the model training function.
            memory (int): Memory (in GB) for the model training function.
            use_gpu (bool): Whether this model needs gpu

        Returns:
            Model: The updated model."""
        return self._call_api('updatePythonModelGit', 'POST', query_params={}, body={'modelId': model_id, 'applicationConnectorId': application_connector_id, 'branchName': branch_name, 'pythonRoot': python_root, 'trainFunctionName': train_function_name, 'predictFunctionName': predict_function_name, 'predictManyFunctionName': predict_many_function_name, 'trainModuleName': train_module_name, 'predictModuleName': predict_module_name, 'trainingInputTables': training_input_tables, 'cpuSize': cpu_size, 'memory': memory, 'useGpu': use_gpu}, parse_type=Model)

    def set_model_training_config(self, model_id: str, training_config: Union[dict, TrainingConfig], feature_group_ids: list = None) -> Model:
        """Edits the default model training config

        Args:
            model_id (str): A unique string identifier of the model to update.
            training_config (TrainingConfig): The training config used to train this model.
            feature_group_ids (list): The list of feature groups used as input to the model.

        Returns:
            Model: The model object corresponding to the updated training config."""
        return self._call_api('setModelTrainingConfig', 'PATCH', query_params={}, body={'modelId': model_id, 'trainingConfig': training_config, 'featureGroupIds': feature_group_ids}, parse_type=Model)

    def set_model_prediction_params(self, model_id: str, prediction_config: dict) -> Model:
        """Sets the model prediction config for the model

        Args:
            model_id (str): Unique string identifier of the model to update.
            prediction_config (dict): Prediction configuration for the model.

        Returns:
            Model: Model object after the prediction configuration is applied."""
        return self._call_api('setModelPredictionParams', 'PATCH', query_params={}, body={'modelId': model_id, 'predictionConfig': prediction_config}, parse_type=Model)

    def retrain_model(self, model_id: str, deployment_ids: list = None, feature_group_ids: list = None, custom_algorithms: list = None, builtin_algorithms: list = None, custom_algorithm_configs: dict = None, cpu_size: str = None, memory: int = None, training_config: Union[dict, TrainingConfig] = None, algorithm_training_configs: list = None) -> Model:
        """Retrains the specified model, with an option to choose the deployments to which the retraining will be deployed.

        Args:
            model_id (str): Unique string identifier of the model to retrain.
            deployment_ids (list): List of unique string identifiers of deployments to automatically deploy to.
            feature_group_ids (list): List of feature group IDs provided by the user to train the model on.
            custom_algorithms (list): List of user-defined algorithms to train. If not set, will honor the runs from the last time and applicable new custom algorithms.
            builtin_algorithms (list): List of the built-in algorithms provided by Abacus.AI to train. If not set, will honor the runs from the last time and applicable new built-in algorithms.
            custom_algorithm_configs (dict): User-defined training configs for each custom algorithm.
            cpu_size (str): Size of the CPU for the user-defined algorithms during training.
            memory (int): Memory (in GB) for the user-defined algorithms during training.
            training_config (TrainingConfig): The training config used to train this model.
            algorithm_training_configs (list): List of algorithm specifc training configs that will be part of the model training AutoML run.

        Returns:
            Model: The model that is being retrained."""
        return self._call_api('retrainModel', 'POST', query_params={}, body={'modelId': model_id, 'deploymentIds': deployment_ids, 'featureGroupIds': feature_group_ids, 'customAlgorithms': custom_algorithms, 'builtinAlgorithms': builtin_algorithms, 'customAlgorithmConfigs': custom_algorithm_configs, 'cpuSize': cpu_size, 'memory': memory, 'trainingConfig': training_config, 'algorithmTrainingConfigs': algorithm_training_configs}, parse_type=Model)

    def delete_model(self, model_id: str):
        """Deletes the specified model and all its versions. Models which are currently used in deployments cannot be deleted.

        Args:
            model_id (str): Unique string identifier of the model to delete."""
        return self._call_api('deleteModel', 'DELETE', query_params={'modelId': model_id})

    def delete_model_version(self, model_version: str):
        """Deletes the specified model version. Model versions which are currently used in deployments cannot be deleted.

        Args:
            model_version (str): The unique identifier of the model version to delete."""
        return self._call_api('deleteModelVersion', 'DELETE', query_params={'modelVersion': model_version})

    def export_model_artifact_as_feature_group(self, model_version: str, table_name: str, artifact_type: str) -> FeatureGroup:
        """Exports metric artifact data for a model as a feature group.

        Args:
            model_version (str): Unique string identifier for the version of the model.
            table_name (str): Name of the feature group table to create.
            artifact_type (str): EvalArtifact enum specifying which artifact to export.

        Returns:
            FeatureGroup: The created feature group."""
        return self._call_api('exportModelArtifactAsFeatureGroup', 'POST', query_params={}, body={'modelVersion': model_version, 'tableName': table_name, 'artifactType': artifact_type}, parse_type=FeatureGroup)

    def get_custom_train_function_info(self, project_id: str, feature_group_names_for_training: list = None, training_data_parameter_name_override: dict = None, training_config: Union[dict, TrainingConfig] = None, custom_algorithm_config: dict = None) -> CustomTrainFunctionInfo:
        """Returns information about how to call the custom train function.

        Args:
            project_id (str): The unique version ID of the project.
            feature_group_names_for_training (list): A list of feature group table names to be used for training.
            training_data_parameter_name_override (dict): Override from feature group type to parameter name in the train function.
            training_config (TrainingConfig): Training config for the options supported by the Abacus.ai platform.
            custom_algorithm_config (dict): User-defined config that can be serialized by JSON.

        Returns:
            CustomTrainFunctionInfo: Information about how to call the customer-provided train function."""
        return self._call_api('getCustomTrainFunctionInfo', 'POST', query_params={}, body={'projectId': project_id, 'featureGroupNamesForTraining': feature_group_names_for_training, 'trainingDataParameterNameOverride': training_data_parameter_name_override, 'trainingConfig': training_config, 'customAlgorithmConfig': custom_algorithm_config}, parse_type=CustomTrainFunctionInfo)

    def create_model_monitor(self, project_id: str, prediction_feature_group_id: str, training_feature_group_id: str = None, name: str = None, refresh_schedule: str = None, target_value: str = None, target_value_bias: str = None, target_value_performance: str = None, feature_mappings: dict = None, model_id: str = None, training_feature_mappings: dict = None, feature_group_base_monitor_config: dict = None, feature_group_comparison_monitor_config: dict = None) -> ModelMonitor:
        """Runs a model monitor for the specified project.

        Args:
            project_id (str): The unique ID associated with the project.
            prediction_feature_group_id (str): The unique ID of the prediction data feature group.
            training_feature_group_id (str): The unique ID of the training data feature group.
            name (str): The name you want your model monitor to have. Defaults to "<Project Name> Model Monitor".
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically retrain the created model monitor.
            target_value (str): A target positive value for the label to compute bias and PR/AUC for performance page.
            target_value_bias (str): A target positive value for the label to compute bias.
            target_value_performance (str): A target positive value for the label to compute PR curve/AUC for performance page.
            feature_mappings (dict): A JSON map to override features for prediction_feature_group, where keys are column names and the values are feature data use types.
            model_id (str): The unique ID of the model.
            training_feature_mappings (dict): A JSON map to override features for training_fature_group, where keys are column names and the values are feature data use types.
            feature_group_base_monitor_config (dict): Selection strategy for the feature_group 1 with the feature group version if selected.
            feature_group_comparison_monitor_config (dict): Selection strategy for the feature_group 1 with the feature group version if selected.

        Returns:
            ModelMonitor: The new model monitor that was created."""
        return self._call_api('createModelMonitor', 'POST', query_params={}, body={'projectId': project_id, 'predictionFeatureGroupId': prediction_feature_group_id, 'trainingFeatureGroupId': training_feature_group_id, 'name': name, 'refreshSchedule': refresh_schedule, 'targetValue': target_value, 'targetValueBias': target_value_bias, 'targetValuePerformance': target_value_performance, 'featureMappings': feature_mappings, 'modelId': model_id, 'trainingFeatureMappings': training_feature_mappings, 'featureGroupBaseMonitorConfig': feature_group_base_monitor_config, 'featureGroupComparisonMonitorConfig': feature_group_comparison_monitor_config}, parse_type=ModelMonitor)

    def rerun_model_monitor(self, model_monitor_id: str) -> ModelMonitor:
        """Re-runs the specified model monitor.

        Args:
            model_monitor_id (str): Unique string identifier of the model monitor to re-run.

        Returns:
            ModelMonitor: The model monitor that is being re-run."""
        return self._call_api('rerunModelMonitor', 'POST', query_params={}, body={'modelMonitorId': model_monitor_id}, parse_type=ModelMonitor)

    def rename_model_monitor(self, model_monitor_id: str, name: str):
        """Renames a model monitor

        Args:
            model_monitor_id (str): Unique identifier of the model monitor to rename.
            name (str): The new name to apply to the model monitor."""
        return self._call_api('renameModelMonitor', 'PATCH', query_params={}, body={'modelMonitorId': model_monitor_id, 'name': name})

    def delete_model_monitor(self, model_monitor_id: str):
        """Deletes the specified Model Monitor and all its versions.

        Args:
            model_monitor_id (str): Unique identifier of the Model Monitor to delete."""
        return self._call_api('deleteModelMonitor', 'DELETE', query_params={'modelMonitorId': model_monitor_id})

    def delete_model_monitor_version(self, model_monitor_version: str):
        """Deletes the specified model monitor version.

        Args:
            model_monitor_version (str): Unique identifier of the model monitor version to delete."""
        return self._call_api('deleteModelMonitorVersion', 'DELETE', query_params={'modelMonitorVersion': model_monitor_version})

    def create_vision_drift_monitor(self, project_id: str, prediction_feature_group_id: str, training_feature_group_id: str, name: str, feature_mappings: dict, training_feature_mappings: dict, target_value_performance: str = None, refresh_schedule: str = None) -> ModelMonitor:
        """Runs a vision drift monitor for the specified project.

        Args:
            project_id (str): Unique string identifier of the project.
            prediction_feature_group_id (str): Unique string identifier of the prediction data feature group.
            training_feature_group_id (str): Unique string identifier of the training data feature group.
            name (str): The name you want your model monitor to have. Defaults to "<Project Name> Model Monitor".
            feature_mappings (dict): A JSON map to override features for prediction_feature_group, where keys are column names and the values are feature data use types.
            training_feature_mappings (dict): A JSON map to override features for training_feature_group, where keys are column names and the values are feature data use types.
            target_value_performance (str): A target positive value for the label to compute precision-recall curve/area under curve for performance page.
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically rerun the created vision drift monitor.

        Returns:
            ModelMonitor: The new model monitor that was created."""
        return self._call_api('createVisionDriftMonitor', 'POST', query_params={}, body={'projectId': project_id, 'predictionFeatureGroupId': prediction_feature_group_id, 'trainingFeatureGroupId': training_feature_group_id, 'name': name, 'featureMappings': feature_mappings, 'trainingFeatureMappings': training_feature_mappings, 'targetValuePerformance': target_value_performance, 'refreshSchedule': refresh_schedule}, parse_type=ModelMonitor)

    def create_nlp_drift_monitor(self, project_id: str, prediction_feature_group_id: str, training_feature_group_id: str, name: str, feature_mappings: dict, training_feature_mappings: dict, target_value_performance: str = None, refresh_schedule: str = None) -> ModelMonitor:
        """Runs an NLP drift monitor for the specified project.

        Args:
            project_id (str): Unique string identifier of the project.
            prediction_feature_group_id (str): Unique string identifier of the prediction data feature group.
            training_feature_group_id (str): Unique string identifier of the training data feature group.
            name (str): The name you want your model monitor to have. Defaults to "<Project Name> Model Monitor".
            feature_mappings (dict): A JSON map to override features for prediction_feature_group, where keys are column names and the values are feature data use types.
            training_feature_mappings (dict): A JSON map to override features for training_feature_group, where keys are column names and the values are feature data use types.
            target_value_performance (str): A target positive value for the label to compute precision-recall curve/area under curve for performance page.
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically rerun the created nlp drift monitor.

        Returns:
            ModelMonitor: The new model monitor that was created."""
        return self._call_api('createNlpDriftMonitor', 'POST', query_params={}, body={'projectId': project_id, 'predictionFeatureGroupId': prediction_feature_group_id, 'trainingFeatureGroupId': training_feature_group_id, 'name': name, 'featureMappings': feature_mappings, 'trainingFeatureMappings': training_feature_mappings, 'targetValuePerformance': target_value_performance, 'refreshSchedule': refresh_schedule}, parse_type=ModelMonitor)

    def create_eda(self, project_id: str, feature_group_id: str, name: str, refresh_schedule: str = None, include_collinearity: bool = False, include_data_consistency: bool = False, collinearity_keys: list = None, primary_keys: list = None, data_consistency_test_config: dict = None, data_consistency_reference_config: dict = None, feature_mappings: dict = None, forecast_frequency: str = None) -> Eda:
        """Run an Exploratory Data Analysis (EDA) for the specified project.

        Args:
            project_id (str): The unique ID associated with the project.
            feature_group_id (str): The unique ID of the prediction data feature group.
            name (str): The name you want your model monitor to have. Defaults to "<Project Name> EDA".
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically retrain the created EDA.
            include_collinearity (bool): Set to True if the EDA type is collinearity.
            include_data_consistency (bool): Set to True if the EDA type is data consistency.
            collinearity_keys (list): List of features to use for collinearity
            primary_keys (list): List of features that corresponds to the primary keys or item ids for the given feature group for Data Consistency analysis or Forecasting analysis respectively.
            data_consistency_test_config (dict): Test feature group version selection strategy for Data Consistency EDA type.
            data_consistency_reference_config (dict): Reference feature group version selection strategy for Data Consistency EDA type.
            feature_mappings (dict): A JSON map to override features for the given feature_group, where keys are column names and the values are feature data use types. (In forecasting, used to set the timestamp column and target value)
            forecast_frequency (str): The frequency of the data. It can be either HOURLY, DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY.

        Returns:
            Eda: The new EDA object that was created."""
        return self._call_api('createEda', 'POST', query_params={}, body={'projectId': project_id, 'featureGroupId': feature_group_id, 'name': name, 'refreshSchedule': refresh_schedule, 'includeCollinearity': include_collinearity, 'includeDataConsistency': include_data_consistency, 'collinearityKeys': collinearity_keys, 'primaryKeys': primary_keys, 'dataConsistencyTestConfig': data_consistency_test_config, 'dataConsistencyReferenceConfig': data_consistency_reference_config, 'featureMappings': feature_mappings, 'forecastFrequency': forecast_frequency}, parse_type=Eda)

    def rerun_eda(self, eda_id: str) -> Eda:
        """Reruns the specified EDA object.

        Args:
            eda_id (str): Unique string identifier of the EDA object to rerun.

        Returns:
            Eda: The EDA object that is being rerun."""
        return self._call_api('rerunEda', 'POST', query_params={}, body={'edaId': eda_id}, parse_type=Eda)

    def rename_eda(self, eda_id: str, name: str):
        """Renames an EDA

        Args:
            eda_id (str): Unique string identifier of the EDA to rename.
            name (str): The new name to apply to the model monitor."""
        return self._call_api('renameEda', 'PATCH', query_params={}, body={'edaId': eda_id, 'name': name})

    def delete_eda(self, eda_id: str):
        """Deletes the specified EDA and all its versions.

        Args:
            eda_id (str): Unique string identifier of the EDA to delete."""
        return self._call_api('deleteEda', 'DELETE', query_params={'edaId': eda_id})

    def delete_eda_version(self, eda_version: str):
        """Deletes the specified EDA version.

        Args:
            eda_version (str): Unique string identifier of the EDA version to delete."""
        return self._call_api('deleteEdaVersion', 'DELETE', query_params={'edaVersion': eda_version})

    def create_monitor_alert(self, project_id: str, model_monitor_id: str, alert_name: str, condition_config: dict, action_config: dict) -> MonitorAlert:
        """Create a monitor alert for the given conditions and monitor

        Args:
            project_id (str): Unique string identifier for the project.
            model_monitor_id (str): Unique string identifier for the model monitor created under the project.
            alert_name (str): Name of the alert.
            condition_config (dict): Condition to run the actions for the alert.
            action_config (dict): Configuration for the action of the alert.

        Returns:
            MonitorAlert: Object describing the monitor alert."""
        return self._call_api('createMonitorAlert', 'POST', query_params={}, body={'projectId': project_id, 'modelMonitorId': model_monitor_id, 'alertName': alert_name, 'conditionConfig': condition_config, 'actionConfig': action_config}, parse_type=MonitorAlert)

    def update_monitor_alert(self, monitor_alert_id: str, alert_name: str = None, condition_config: dict = None, action_config: dict = None) -> MonitorAlert:
        """Update monitor alert

        Args:
            monitor_alert_id (str): Unique identifier of the monitor alert.
            alert_name (str): Name of the alert.
            condition_config (dict): Condition to run the actions for the alert.
            action_config (dict): Configuration for the action of the alert.

        Returns:
            MonitorAlert: Object describing the monitor alert."""
        return self._call_api('updateMonitorAlert', 'POST', query_params={}, body={'monitorAlertId': monitor_alert_id, 'alertName': alert_name, 'conditionConfig': condition_config, 'actionConfig': action_config}, parse_type=MonitorAlert)

    def run_monitor_alert(self, monitor_alert_id: str) -> MonitorAlert:
        """Reruns a given monitor alert from latest monitor instance

        Args:
            monitor_alert_id (str): Unique identifier of a monitor alert.

        Returns:
            MonitorAlert: Object describing the monitor alert."""
        return self._call_api('runMonitorAlert', 'POST', query_params={}, body={'monitorAlertId': monitor_alert_id}, parse_type=MonitorAlert)

    def delete_monitor_alert(self, monitor_alert_id: str):
        """Delets a monitor alert

        Args:
            monitor_alert_id (str): The unique string identifier of the alert to delete."""
        return self._call_api('deleteMonitorAlert', 'DELETE', query_params={'monitorAlertId': monitor_alert_id})

    def create_deployment(self, name: str = None, model_id: str = None, model_version: str = None, algorithm: str = None, additional_model_name: str = None, feature_group_id: str = None, project_id: str = None, description: str = None, calls_per_second: int = None, auto_deploy: bool = True, start: bool = True, enable_batch_streaming_updates: bool = False, model_deployment_config: dict = None, deployment_config: dict = None) -> Deployment:
        """Creates a deployment with the specified name and description for the specified model or feature group.

        A Deployment makes the trained model or feature group available for prediction requests.


        Args:
            name (str): The name of the deployment.
            model_id (str): The unique ID associated with the model.
            model_version (str): The unique ID associated with the model version to deploy.
            algorithm (str): The unique ID associated with the algorithm to deploy.
            additional_model_name (str): The unique name associated with the additional model to deploy.
            feature_group_id (str): The unique ID associated with a feature group.
            project_id (str): The unique ID associated with a project.
            description (str): The description for the deployment.
            calls_per_second (int): The number of calls per second the deployment can handle.
            auto_deploy (bool): Flag to enable the automatic deployment when a new Model Version finishes training.
            start (bool): If true, will start the deployment; otherwise will create offline
            enable_batch_streaming_updates (bool): Flag to enable marking the feature group deployment to have a background process cache streamed in rows for quicker lookup.
            model_deployment_config (dict): The deployment config for model to deploy
            deployment_config (dict): Additional parameters specific to different use_cases

        Returns:
            Deployment: The new model or feature group deployment."""
        return self._call_api('createDeployment', 'POST', query_params={}, body={'name': name, 'modelId': model_id, 'modelVersion': model_version, 'algorithm': algorithm, 'additionalModelName': additional_model_name, 'featureGroupId': feature_group_id, 'projectId': project_id, 'description': description, 'callsPerSecond': calls_per_second, 'autoDeploy': auto_deploy, 'start': start, 'enableBatchStreamingUpdates': enable_batch_streaming_updates, 'modelDeploymentConfig': model_deployment_config, 'deploymentConfig': deployment_config}, parse_type=Deployment)

    def create_deployment_token(self, project_id: str, name: str = None) -> DeploymentAuthToken:
        """Creates a deployment token for the specified project.

        Deployment tokens are used to authenticate requests to the prediction APIs and are scoped to the project level.


        Args:
            project_id (str): The unique string identifier associated with the project.
            name (str): The name of the deployment token.

        Returns:
            DeploymentAuthToken: The deployment token."""
        return self._call_api('createDeploymentToken', 'POST', query_params={}, body={'projectId': project_id, 'name': name}, parse_type=DeploymentAuthToken)

    def update_deployment(self, deployment_id: str, description: str = None):
        """Updates a deployment's description.

        Args:
            deployment_id (str): Unique identifier of the deployment to update.
            description (str): The new description for the deployment."""
        return self._call_api('updateDeployment', 'PATCH', query_params={'deploymentId': deployment_id}, body={'description': description})

    def rename_deployment(self, deployment_id: str, name: str):
        """Updates a deployment's name

        Args:
            deployment_id (str): Unique string identifier for the deployment to update.
            name (str): The new deployment name."""
        return self._call_api('renameDeployment', 'PATCH', query_params={'deploymentId': deployment_id}, body={'name': name})

    def set_auto_deployment(self, deployment_id: str, enable: bool = None):
        """Enable or disable auto deployment for the specified deployment.

        When a model is scheduled to retrain, deployments with auto deployment enabled will be marked to automatically promote the new model version. After the newly trained model completes, a check on its metrics in comparison to the currently deployed model version will be performed. If the metrics are comparable or better, the newly trained model version is automatically promoted. If not, it will be marked as a failed model version promotion with an error indicating poor metrics performance.


        Args:
            deployment_id (str): The unique ID associated with the deployment.
            enable (bool): Enable or disable the autoDeploy property of the deployment."""
        return self._call_api('setAutoDeployment', 'POST', query_params={'deploymentId': deployment_id}, body={'enable': enable})

    def set_deployment_model_version(self, deployment_id: str, model_version: str, algorithm: str = None, model_deployment_config: dict = None):
        """Promotes a model version and/or algorithm to be the active served deployment version

        Args:
            deployment_id (str): A unique identifier for the deployment.
            model_version (str): A unique identifier for the model version.
            algorithm (str): The algorithm to use for the model version. If not specified, the algorithm will be inferred from the model version.
            model_deployment_config (dict): The deployment configuration for the model to deploy."""
        return self._call_api('setDeploymentModelVersion', 'PATCH', query_params={'deploymentId': deployment_id}, body={'modelVersion': model_version, 'algorithm': algorithm, 'modelDeploymentConfig': model_deployment_config})

    def set_deployment_feature_group_version(self, deployment_id: str, feature_group_version: str):
        """Promotes a feature group version to be served in the deployment.

        Args:
            deployment_id (str): Unique string identifier for the deployment.
            feature_group_version (str): Unique string identifier for the feature group version."""
        return self._call_api('setDeploymentFeatureGroupVersion', 'PATCH', query_params={'deploymentId': deployment_id}, body={'featureGroupVersion': feature_group_version})

    def start_deployment(self, deployment_id: str):
        """Restarts the specified deployment that was previously suspended.

        Args:
            deployment_id (str): A unique string identifier associated with the deployment."""
        return self._call_api('startDeployment', 'POST', query_params={'deploymentId': deployment_id}, body={})

    def stop_deployment(self, deployment_id: str):
        """Stops the specified deployment.

        Args:
            deployment_id (str): Unique string identifier of the deployment to be stopped."""
        return self._call_api('stopDeployment', 'POST', query_params={'deploymentId': deployment_id}, body={})

    def delete_deployment(self, deployment_id: str):
        """Deletes the specified deployment. The deployment's models will not be affected. Note that the deployments are not recoverable after they are deleted.

        Args:
            deployment_id (str): Unique string identifier of the deployment to delete."""
        return self._call_api('deleteDeployment', 'DELETE', query_params={'deploymentId': deployment_id})

    def delete_deployment_token(self, deployment_token: str):
        """Deletes the specified deployment token.

        Args:
            deployment_token (str): The deployment token to delete."""
        return self._call_api('deleteDeploymentToken', 'DELETE', query_params={'deploymentToken': deployment_token})

    def set_deployment_feature_group_export_file_connector_output(self, deployment_id: str, file_format: str = None, output_location: str = None):
        """Sets the export output for the Feature Group Deployment to be a file connector.

        Args:
            deployment_id (str): The ID of the deployment for which the export type is set.
            file_format (str): The type of export output, either CSV or JSON.
            output_location (str): The file connector (cloud) location where the output should be exported."""
        return self._call_api('setDeploymentFeatureGroupExportFileConnectorOutput', 'POST', query_params={'deploymentId': deployment_id}, body={'fileFormat': file_format, 'outputLocation': output_location})

    def set_deployment_feature_group_export_database_connector_output(self, deployment_id: str, database_connector_id: str, object_name: str, write_mode: str, database_feature_mapping: dict, id_column: str = None, additional_id_columns: list = None):
        """Sets the export output for the Feature Group Deployment to a Database connector.

        Args:
            deployment_id (str): The ID of the deployment for which the export type is set.
            database_connector_id (str): The unique string identifier of the database connector used.
            object_name (str): The object of the database connector to write to.
            write_mode (str): The write mode to use when writing to the database connector, either UPSERT or INSERT.
            database_feature_mapping (dict): The column/feature pairs mapping the features to the database columns.
            id_column (str): The id column to use as the upsert key.
            additional_id_columns (list): For database connectors which support it, a list of additional ID columns to use as a complex key for upserting."""
        return self._call_api('setDeploymentFeatureGroupExportDatabaseConnectorOutput', 'POST', query_params={'deploymentId': deployment_id}, body={'databaseConnectorId': database_connector_id, 'objectName': object_name, 'writeMode': write_mode, 'databaseFeatureMapping': database_feature_mapping, 'idColumn': id_column, 'additionalIdColumns': additional_id_columns})

    def remove_deployment_feature_group_export_output(self, deployment_id: str):
        """Removes the export type that is set for the Feature Group Deployment

        Args:
            deployment_id (str): The ID of the deployment for which the export type is set."""
        return self._call_api('removeDeploymentFeatureGroupExportOutput', 'POST', query_params={'deploymentId': deployment_id}, body={})

    def create_refresh_policy(self, name: str, cron: str, refresh_type: str, project_id: str = None, dataset_ids: list = [], feature_group_id: str = None, model_ids: list = [], deployment_ids: list = [], batch_prediction_ids: list = [], prediction_metric_ids: list = [], model_monitor_ids: list = [], notebook_id: str = None, feature_group_export_config: Union[dict, FeatureGroupExportConfig] = None) -> RefreshPolicy:
        """Creates a refresh policy with a particular cron pattern and refresh type.

        A refresh policy allows for the scheduling of a set of actions at regular intervals. This can be useful for periodically updating data that needs to be re-imported into the project for retraining.


        Args:
            name (str): The name of the refresh policy.
            cron (str): A cron-like string specifying the frequency of the refresh policy.
            refresh_type (str): The refresh type used to determine what is being refreshed, such as a single dataset, dataset and model, or more.
            project_id (str): Optionally, a project ID can be specified so that all datasets, models, deployments, batch predictions, prediction metrics, model monitrs, and notebooks are captured at the instant the policy was created.
            dataset_ids (list): Comma-separated list of dataset IDs.
            feature_group_id (str): Feature Group ID associated with refresh policy.
            model_ids (list): Comma-separated list of model IDs.
            deployment_ids (list): Comma-separated list of deployment IDs.
            batch_prediction_ids (list): Comma-separated list of batch prediction IDs.
            prediction_metric_ids (list): Comma-separated list of prediction metric IDs.
            model_monitor_ids (list): Comma-separated list of model monitor IDs.
            notebook_id (str): Notebook ID associated with refresh policy.
            feature_group_export_config (FeatureGroupExportConfig): Feature group export configuration.

        Returns:
            RefreshPolicy: The created refresh policy."""
        return self._call_api('createRefreshPolicy', 'POST', query_params={}, body={'name': name, 'cron': cron, 'refreshType': refresh_type, 'projectId': project_id, 'datasetIds': dataset_ids, 'featureGroupId': feature_group_id, 'modelIds': model_ids, 'deploymentIds': deployment_ids, 'batchPredictionIds': batch_prediction_ids, 'predictionMetricIds': prediction_metric_ids, 'modelMonitorIds': model_monitor_ids, 'notebookId': notebook_id, 'featureGroupExportConfig': feature_group_export_config}, parse_type=RefreshPolicy)

    def delete_refresh_policy(self, refresh_policy_id: str):
        """Delete a refresh policy.

        Args:
            refresh_policy_id (str): Unique string identifier associated with the refresh policy to delete."""
        return self._call_api('deleteRefreshPolicy', 'DELETE', query_params={'refreshPolicyId': refresh_policy_id})

    def pause_refresh_policy(self, refresh_policy_id: str):
        """Pauses a refresh policy

        Args:
            refresh_policy_id (str): Unique identifier associated with the refresh policy to be paused."""
        return self._call_api('pauseRefreshPolicy', 'POST', query_params={}, body={'refreshPolicyId': refresh_policy_id})

    def resume_refresh_policy(self, refresh_policy_id: str):
        """Resumes a refresh policy

        Args:
            refresh_policy_id (str): The unique ID associated with this refresh policy."""
        return self._call_api('resumeRefreshPolicy', 'POST', query_params={}, body={'refreshPolicyId': refresh_policy_id})

    def run_refresh_policy(self, refresh_policy_id: str):
        """Force a run of the refresh policy.

        Args:
            refresh_policy_id (str): Unique string identifier associated with the refresh policy to be run."""
        return self._call_api('runRefreshPolicy', 'POST', query_params={}, body={'refreshPolicyId': refresh_policy_id})

    def update_refresh_policy(self, refresh_policy_id: str, name: str = None, cron: str = None, feature_group_export_config: Union[dict, FeatureGroupExportConfig] = None) -> RefreshPolicy:
        """Update the name or cron string of a refresh policy

        Args:
            refresh_policy_id (str): Unique string identifier associated with the refresh policy.
            name (str): Name of the refresh policy to be updated.
            cron (str): Cron string describing the schedule from the refresh policy to be updated.
            feature_group_export_config (FeatureGroupExportConfig): Feature group export configuration to update a feature group refresh policy.

        Returns:
            RefreshPolicy: Updated refresh policy."""
        return self._call_api('updateRefreshPolicy', 'POST', query_params={}, body={'refreshPolicyId': refresh_policy_id, 'name': name, 'cron': cron, 'featureGroupExportConfig': feature_group_export_config}, parse_type=RefreshPolicy)

    def lookup_features(self, deployment_token: str, deployment_id: str, query_data: dict, limit_results: int = None, result_columns: list = None) -> Dict:
        """Returns the feature group deployed in the feature store project.

        Args:
            deployment_token (str): A deployment token used to authenticate access to created deployments. This token only authorizes predictions on deployments in this project, so it can be safely embedded inside an application or website.
            deployment_id (str): A unique identifier for a deployment created under the project.
            query_data (dict): A dictionary where the key is the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and the value is the unique value of the same entity.
            limit_results (int): If provided, will limit the number of results to the value specified.
            result_columns (list): If provided, will limit the columns present in each result to the columns specified in this list."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('lookupFeatures', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'limitResults': limit_results, 'resultColumns': result_columns}, server_override=prediction_url)

    def predict(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns a prediction for Predictive Modeling

        Args:
            deployment_token (str): A deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, and is safe to embed in an application or website.
            deployment_id (str): A unique identifier for a deployment created under the project.
            query_data (dict): A dictionary where the key is the column name (e.g. a column with name 'user_id' in the dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed, and the value is the unique value of the same entity."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('predict', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def predict_multiple(self, deployment_token: str, deployment_id: str, query_data: list) -> Dict:
        """Returns a list of predictions for predictive modeling.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, and is safe to embed in an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            query_data (list): A list of dictionaries, where the 'key' is the column name (e.g. a column with name 'user_id' in the dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed, and the 'value' is the unique value of the same entity."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('predictMultiple', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def predict_from_datasets(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns a list of predictions for Predictive Modeling.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            query_data (dict): A dictionary where the 'key' is the source dataset name, and the 'value' is a list of records corresponding to the dataset rows."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('predictFromDatasets', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def predict_lead(self, deployment_token: str, deployment_id: str, query_data: dict, explain_predictions: bool = False, explainer_type: str = None) -> Dict:
        """Returns the probability of a user being a lead based on their interaction with the service/product and their own attributes (e.g. income, assets, credit score, etc.). Note that the inputs to this method, wherever applicable, should be the column names in the dataset mapped to the column mappings in our system (e.g. column 'user_id' mapped to mapping 'LEAD_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): A dictionary containing user attributes and/or user's interaction data with the product/service (e.g. number of clicks, items in cart, etc.).
            explain_predictions (bool): Will explain predictions for leads
            explainer_type (str): Type of explainer to use for explanations"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('predictLead', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'explainPredictions': explain_predictions, 'explainerType': explainer_type}, server_override=prediction_url)

    def predict_churn(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns the probability of a user to churn out in response to their interactions with the item/product/service. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'churn_result' mapped to mapping 'CHURNED_YN' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where the 'key' will be the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and the 'value' will be the unique value of the same entity."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('predictChurn', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def predict_takeover(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns a probability for each class label associated with the types of fraud or a 'yes' or 'no' type label for the possibility of fraud. Note that the inputs to this method, wherever applicable, will be the column names in the dataset mapped to the column mappings in our system (e.g., column 'account_name' mapped to mapping 'ACCOUNT_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): A dictionary containing account activity characteristics (e.g., login id, login duration, login type, IP address, etc.)."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('predictTakeover', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def predict_fraud(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns the probability of a transaction performed under a specific account being fraudulent or not. Note that the inputs to this method, wherever applicable, should be the column names in your dataset mapped to the column mappings in our system (e.g. column 'account_number' mapped to the mapping 'ACCOUNT_ID' in our system).

        Args:
            deployment_token (str): A deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique identifier to a deployment created under the project.
            query_data (dict): A dictionary containing transaction attributes (e.g. credit card type, transaction location, transaction amount, etc.)."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('predictFraud', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def predict_class(self, deployment_token: str, deployment_id: str, query_data: dict, threshold: float = None, threshold_class: str = None, thresholds: list = None, explain_predictions: bool = False, fixed_features: list = None, nested: str = None, explainer_type: str = None) -> Dict:
        """Returns a classification prediction

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model within an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            query_data (dict): A dictionary where the 'Key' is the column name (e.g. a column with the name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and the 'Value' is the unique value of the same entity.
            threshold (float): A float value that is applied on the popular class label.
            threshold_class (str): The label upon which the threshold is added (binary labels only).
            thresholds (list): Maps labels to thresholds (multi-label classification only). Defaults to F1 optimal threshold if computed for the given class, else uses 0.5.
            explain_predictions (bool): If True, returns the SHAP explanations for all input features.
            fixed_features (list): A set of input features to treat as constant for explanations.
            nested (str): If specified generates prediction delta for each index of the specified nested feature.
            explainer_type (str): The type of explainer to use."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('predictClass', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'threshold': threshold, 'thresholdClass': threshold_class, 'thresholds': thresholds, 'explainPredictions': explain_predictions, 'fixedFeatures': fixed_features, 'nested': nested, 'explainerType': explainer_type}, server_override=prediction_url)

    def predict_target(self, deployment_token: str, deployment_id: str, query_data: dict, explain_predictions: bool = False, fixed_features: list = None, nested: str = None, explainer_type: str = None) -> Dict:
        """Returns a prediction from a classification or regression model. Optionally, includes explanations.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            query_data (dict): A dictionary where the 'key' is the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and the 'value' is the unique value of the same entity.
            explain_predictions (bool): If true, returns the SHAP explanations for all input features.
            fixed_features (list): Set of input features to treat as constant for explanations.
            nested (str): If specified, generates prediction delta for each index of the specified nested feature.
            explainer_type (str): The type of explainer to use."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('predictTarget', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'explainPredictions': explain_predictions, 'fixedFeatures': fixed_features, 'nested': nested, 'explainerType': explainer_type}, server_override=prediction_url)

    def get_anomalies(self, deployment_token: str, deployment_id: str, threshold: float = None, histogram: bool = False) -> io.BytesIO:
        """Returns a list of anomalies from the training dataset.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            threshold (float): The threshold score of what is an anomaly. Valid values are between 0.8 and 0.99.
            histogram (bool): If True, will return a histogram of the distribution of all points."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getAnomalies', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'threshold': threshold, 'histogram': histogram}, server_override=prediction_url)

    def is_anomaly(self, deployment_token: str, deployment_id: str, query_data: dict = None) -> Dict:
        """Returns a list of anomaly attributes based on login information for a specified account. Note that the inputs to this method, wherever applicable, should be the column names in the dataset mapped to the column mappings in our system (e.g. column 'account_name' mapped to mapping 'ACCOUNT_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): The input data for the prediction."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('isAnomaly', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def get_forecast(self, deployment_token: str, deployment_id: str, query_data: dict, future_data: list = None, num_predictions: int = None, prediction_start: str = None, explain_predictions: bool = False, explainer_type: str = None) -> Dict:
        """Returns a list of forecasts for a given entity under the specified project deployment. Note that the inputs to the deployed model will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'holiday_yn' mapped to mapping 'FUTURE' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where 'Key' will be the column name (e.g. a column with name 'store_id' in your dataset) mapped to the column mapping ITEM_ID that uniquely identifies the entity against which forecasting is performed and 'Value' will be the unique value of the same entity.
            future_data (list): This will be a list of values known ahead of time that are relevant for forecasting (e.g. State Holidays, National Holidays, etc.). Each element is a dictionary, where the key and the value both will be of type 'str'. For example future data entered for a Store may be [{"Holiday":"No", "Promo":"Yes", "Date": "2015-07-31 00:00:00"}].
            num_predictions (int): The number of timestamps to predict in the future.
            prediction_start (str): The start date for predictions (e.g., "2015-08-01T00:00:00" as input for mid-night of 2015-08-01).
            explain_predictions (bool): Will explain predictions for forecasting
            explainer_type (str): Type of explainer to use for explanations"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getForecast', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'futureData': future_data, 'numPredictions': num_predictions, 'predictionStart': prediction_start, 'explainPredictions': explain_predictions, 'explainerType': explainer_type}, server_override=prediction_url)

    def get_k_nearest(self, deployment_token: str, deployment_id: str, vector: list, k: int = None, distance: str = None, include_score: bool = False, catalog_id: str = None) -> Dict:
        """Returns the k nearest neighbors for the provided embedding vector.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            vector (list): Input vector to perform the k nearest neighbors with.
            k (int): Overrideable number of items to return.
            distance (str): Specify the distance function to use when finding nearest neighbors.
            include_score (bool): If True, will return the score alongside the resulting embedding value.
            catalog_id (str): An optional parameter honored only for embeddings that provide a catalog id"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getKNearest', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'vector': vector, 'k': k, 'distance': distance, 'includeScore': include_score, 'catalogId': catalog_id}, server_override=prediction_url)

    def get_multiple_k_nearest(self, deployment_token: str, deployment_id: str, queries: list):
        """Returns the k nearest neighbors for the queries provided.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            queries (list): List of mappings of format {"catalogId": "cat0", "vectors": [...], "k": 20, "distance": "euclidean"}. See `getKNearest` for additional information about the supported parameters."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getMultipleKNearest', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queries': queries}, server_override=prediction_url)

    def get_labels(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns a list of scored labels for a document.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): Dictionary where key is "Content" and value is the text from which entities are to be extracted."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getLabels', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def get_entities_from_pdf(self, deployment_token: str, deployment_id: str, pdf: io.TextIOBase = None, doc_id: str = None, return_extracted_features: bool = False, verbose: bool = False) -> Dict:
        """Extracts text from the provided PDF and returns a list of recognized labels and their scores.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            pdf (io.TextIOBase): (Optional) The pdf to predict on. One of pdf or docId must be specified.
            doc_id (str): (Optional) The pdf to predict on. One of pdf or docId must be specified.
            return_extracted_features (bool): (Optional) If True, will return all extracted features (e.g. all tokens in a page) from the PDF. Default is False.
            verbose (bool): (Optional) If True, will return all the extracted tokens probabilities for all the trained labels. Default is False."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getEntitiesFromPDF', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id, 'docId': doc_id, 'returnExtractedFeatures': return_extracted_features, 'verbose': verbose}, files={'pdf': pdf}, server_override=prediction_url)

    def get_recommendations(self, deployment_token: str, deployment_id: str, query_data: dict, num_items: int = None, page: int = None, exclude_item_ids: list = None, score_field: str = None, scaling_factors: list = None, restrict_items: list = None, exclude_items: list = None, explore_fraction: float = None, diversity_attribute_name: str = None, diversity_max_results_per_value: int = None) -> Dict:
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
            restrict_items (list): It allows you to restrict the recommendations to certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1", "value3", ...]}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1", "value3", ...]" to which to restrict the recommendations to. Let's take an example where the input to restrict_items is [{"column": "VehicleType", "values": ["SUV", "Sedan"]}]. This input will restrict the recommendations to SUVs and Sedans. This type of restriction is particularly useful if there's a list of items that you know is of use in some particular scenario and you want to restrict the recommendations only to that list.
            exclude_items (list): It allows you to exclude certain items from the list of recommendations. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1", ...]}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" to exclude from the recommendations. Let's take an example where the input to exclude_items is [{"column": "VehicleType", "values": ["SUV", "Sedan"]}]. The resulting recommendation list will exclude all SUVs and Sedans. This is
            explore_fraction (float): Explore fraction.
            diversity_attribute_name (str): item attribute column name which is used to ensure diversity of prediction results.
            diversity_max_results_per_value (int): maximum number of results per value of diversity_attribute_name."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getRecommendations', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'numItems': num_items, 'page': page, 'excludeItemIds': exclude_item_ids, 'scoreField': score_field, 'scalingFactors': scaling_factors, 'restrictItems': restrict_items, 'excludeItems': exclude_items, 'exploreFraction': explore_fraction, 'diversityAttributeName': diversity_attribute_name, 'diversityMaxResultsPerValue': diversity_max_results_per_value}, server_override=prediction_url)

    def get_personalized_ranking(self, deployment_token: str, deployment_id: str, query_data: dict, preserve_ranks: list = None, preserve_unknown_items: bool = False, scaling_factors: list = None) -> Dict:
        """Returns a list of items with personalized promotions for a given user under the specified project deployment. Note that the inputs to this method, wherever applicable, should be the column names in the dataset mapped to the column mappings in our system (e.g. column 'item_code' mapped to mapping 'ITEM_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model in an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This should be a dictionary with two key-value pairs. The first pair represents a 'Key' where the column name (e.g. a column with name 'user_id' in the dataset) mapped to the column mapping USER_ID uniquely identifies the user against whom a prediction is made and a 'Value' which is the identifier value for that user. The second pair will have a 'Key' which will be the name of the column name (e.g. movie_name) mapped to ITEM_ID (unique item identifier) and a 'Value' which will be a list of identifiers that uniquely identifies those items.
            preserve_ranks (list): List of dictionaries of format {"column": "col0", "values": ["value0, value1"]}, where the ranks of items in query_data is preserved for all the items in "col0" with values, "value0" and "value1". This option is useful when the desired items are being recommended in the desired order and the ranks for those items need to be kept unchanged during recommendation generation.
            preserve_unknown_items (bool): If true, any items that are unknown to the model, will not be reranked, and the original position in the query will be preserved.
            scaling_factors (list): It allows you to bias the model towards certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1"], "factor": 1.1}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" in reference to which the model recommendations need to be biased; and the key, "factor" takes the factor by which the item scores are adjusted. Let's take an example where the input to scaling_factors is [{"column": "VehicleType", "values": ["SUV", "Sedan"], "factor": 1.4}]. After we apply the model to get item probabilities, for every SUV and Sedan in the list, we will multiply the respective probability by 1.1 before sorting. This is particularly useful if there's a type of item that might be less popular but you want to promote it or there's an item that always comes up and you want to demote it."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getPersonalizedRanking', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'preserveRanks': preserve_ranks, 'preserveUnknownItems': preserve_unknown_items, 'scalingFactors': scaling_factors}, server_override=prediction_url)

    def get_ranked_items(self, deployment_token: str, deployment_id: str, query_data: dict, preserve_ranks: list = None, preserve_unknown_items: bool = False, score_field: str = None, scaling_factors: list = None, diversity_attribute_name: str = None, diversity_max_results_per_value: int = None) -> Dict:
        """Returns a list of re-ranked items for a selected user when a list of items is required to be reranked according to the user's preferences. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'item_code' mapped to mapping 'ITEM_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary with two key-value pairs. The first pair represents a 'Key' where the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID uniquely identifies the user against whom a prediction is made and a 'Value' which is the identifier value for that user. The second pair will have a 'Key' which will be the name of the column name (e.g. movie_name) mapped to ITEM_ID (unique item identifier) and a 'Value' which will be a list of identifiers that uniquely identifies those items.
            preserve_ranks (list): List of dictionaries of format {"column": "col0", "values": ["value0, value1"]}, where the ranks of items in query_data is preserved for all the items in "col0" with values, "value0" and "value1". This option is useful when the desired items are being recommended in the desired order and the ranks for those items need to be kept unchanged during recommendation generation.
            preserve_unknown_items (bool): If true, any items that are unknown to the model, will not be reranked, and the original position in the query will be preserved
            score_field (str): The relative item scores are returned in a separate field named with the same name as the key (score_field) for this argument.
            scaling_factors (list): It allows you to bias the model towards certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1"], "factor": 1.1}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" in reference to which the model recommendations need to be biased; and the key, "factor" takes the factor by which the item scores are adjusted. Let's take an example where the input to scaling_factors is [{"column": "VehicleType", "values": ["SUV", "Sedan"], "factor": 1.4}]. After we apply the model to get item probabilities, for every SUV and Sedan in the list, we will multiply the respective probability by 1.1 before sorting. This is particularly useful if there is a type of item that might be less popular but you want to promote it or there is an item that always comes up and you want to demote it.
            diversity_attribute_name (str): item attribute column name which is used to ensure diversity of prediction results.
            diversity_max_results_per_value (int): maximum number of results per value of diversity_attribute_name."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getRankedItems', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'preserveRanks': preserve_ranks, 'preserveUnknownItems': preserve_unknown_items, 'scoreField': score_field, 'scalingFactors': scaling_factors, 'diversityAttributeName': diversity_attribute_name, 'diversityMaxResultsPerValue': diversity_max_results_per_value}, server_override=prediction_url)

    def get_related_items(self, deployment_token: str, deployment_id: str, query_data: dict, num_items: int = None, page: int = None, scaling_factors: list = None, restrict_items: list = None, exclude_items: list = None) -> Dict:
        """Returns a list of related items for a given item under the specified project deployment. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'item_code' mapped to mapping 'ITEM_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where the 'key' will be the column name (e.g. a column with name 'user_name' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the user against which related items are determined and the 'value' will be the unique value of the same item. For example, if you have the column name 'user_name' mapped to the column mapping 'USER_ID', then the query must have the exact same column name (user_name) as key and the name of the user (John Doe) as value.
            num_items (int): The number of items to recommend on one page. By default, it is set to 50 items per page.
            page (int): The page number to be displayed. For example, let's say that the num_items is set to 10 with the total recommendations list size of 50 recommended items, then an input value of 2 in the 'page' variable will display a list of items that rank from 11th to 20th.
            scaling_factors (list): It allows you to bias the model towards certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1"], "factor": 1.1}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" in reference to which the model recommendations need to be biased; and the key, "factor" takes the factor by which the item scores are adjusted.  Let's take an example where the input to scaling_factors is [{"column": "VehicleType", "values": ["SUV", "Sedan"], "factor": 1.4}]. After we apply the model to get item probabilities, for every SUV and Sedan in the list, we will multiply the respective probability by 1.1 before sorting. This is particularly useful if there's a type of item that might be less popular but you want to promote it or there's an item that always comes up and you want to demote it.
            restrict_items (list): It allows you to restrict the recommendations to certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1", "value3", ...]}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1", "value3", ...]" to which to restrict the recommendations to. Let's take an example where the input to restrict_items is [{"column": "VehicleType", "values": ["SUV", "Sedan"]}]. This input will restrict the recommendations to SUVs and Sedans. This type of restriction is particularly useful if there's a list of items that you know is of use in some particular scenario and you want to restrict the recommendations only to that list.
            exclude_items (list): It allows you to exclude certain items from the list of recommendations. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1", ...]}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" to exclude from the recommendations. Let's take an example where the input to exclude_items is [{"column": "VehicleType", "values": ["SUV", "Sedan"]}]. The resulting recommendation list will exclude all SUVs and Sedans. This is particularly useful if there's a list of items that you know is of no use in some particular scenario and you don't want to show those items present in that list."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getRelatedItems', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'numItems': num_items, 'page': page, 'scalingFactors': scaling_factors, 'restrictItems': restrict_items, 'excludeItems': exclude_items}, server_override=prediction_url)

    def get_chat_response(self, deployment_token: str, deployment_id: str, messages: list, llm_name: str = None, num_completion_tokens: int = None, system_message: str = None, temperature: float = None, filter_key_values: dict = None) -> Dict:
        """Return a chat response which continues the conversation based on the input messages and search results.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            messages (list): A list of chronologically ordered messages, starting with a user message and alternating sources. A message is a dict with attributes:     is_user (bool): Whether the message is from the user.      text (str): The message's text.
            llm_name (str): Name of the specific LLM backend to use to power the chat experience
            num_completion_tokens (int): Default for maximum number of tokens for chat answers
            system_message (str): The generative LLM system message
            temperature (float): The generative LLM temperature
            filter_key_values (dict): A dictionary mapping column names to a list of values to restrict the retrived search results."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getChatResponse', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'messages': messages, 'llmName': llm_name, 'numCompletionTokens': num_completion_tokens, 'systemMessage': system_message, 'temperature': temperature, 'filterKeyValues': filter_key_values}, server_override=prediction_url)

    def get_conversation_response(self, deployment_id: str, message: str, deployment_conversation_id: str = None, chat_config: dict = None, filter_key_values: dict = None) -> Dict:
        """Return a conversation response which continues the conversation based on the input message and deployment conversation id (if exists).

        Args:
            deployment_id (str): The unique identifier to a deployment created under the project.
            message (str): A message from the user
            deployment_conversation_id (str): The unique identifier of a deployment conversation to continue. If not specified, only a single response will be returned.
            chat_config (dict): A dictionary specifiying the query chat config override.
            filter_key_values (dict): A dictionary mapping column names to a list of values to restrict the retrived search results."""
        return self._call_api('getConversationResponse', 'POST', query_params={'deploymentId': deployment_id}, body={'message': message, 'deploymentConversationId': deployment_conversation_id, 'chatConfig': chat_config, 'filterKeyValues': filter_key_values})

    def get_search_results(self, deployment_token: str, deployment_id: str, query_data: dict, num: int = 15) -> Dict:
        """Return the most relevant search results to the search query from the uploaded documents.

        Args:
            deployment_token (str): A token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it can be securely embedded in an application or website.
            deployment_id (str): A unique identifier of a deployment created under the project.
            query_data (dict): A dictionary where the key is "Content" and the value is the text from which entities are to be extracted.
            num (int): Number of search results to return."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getSearchResults', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'num': num}, server_override=prediction_url)

    def get_sentiment(self, deployment_token: str, deployment_id: str, document: str) -> Dict:
        """Predicts sentiment on a document

        Args:
            deployment_token (str): A token used to authenticate access to deployments created in this project. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for a deployment created under this project.
            document (str): The document to be analyzed for sentiment."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getSentiment', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'document': document}, server_override=prediction_url)

    def get_entailment(self, deployment_token: str, deployment_id: str, document: str) -> Dict:
        """Predicts the classification of the document

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            document (str): The document to be classified."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getEntailment', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'document': document}, server_override=prediction_url)

    def get_classification(self, deployment_token: str, deployment_id: str, document: str) -> Dict:
        """Predicts the classification of the document

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            document (str): The document to be classified."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getClassification', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'document': document}, server_override=prediction_url)

    def get_summary(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns a JSON of the predicted summary for the given document. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'text' mapped to mapping 'DOCUMENT' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): Raw data dictionary containing the required document data - must have a key 'document' corresponding to a DOCUMENT type text as value."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getSummary', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def predict_language(self, deployment_token: str, deployment_id: str, query_data: str) -> Dict:
        """Predicts the language of the text

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments within this project, making it safe to embed this model in an application or website.
            deployment_id (str): A unique string identifier for a deployment created under the project.
            query_data (str): The input string to detect."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('predictLanguage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def get_assignments(self, deployment_token: str, deployment_id: str, query_data: dict, forced_assignments: dict = None) -> Dict:
        """Get all positive assignments that match a query.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it can be safely embedded in an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            query_data (dict): Specifies the set of assignments being requested. The value for the key can be: 1. A simple scalar value, which is matched exactly 2. A list of values, which matches any element in the list 3. A dictionary with keys lower_in/lower_ex and upper_in/upper_ex, which matches values in an inclusive/exclusive range
            forced_assignments (dict): Set of assignments to force and resolve before returning query results."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getAssignments', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'forcedAssignments': forced_assignments}, server_override=prediction_url)

    def get_alternative_assignments(self, deployment_token: str, deployment_id: str, query_data: dict, add_constraints: list = None) -> Dict:
        """Get alternative positive assignments for given query. Optimal assignments are ignored and the alternative assignments are returned instead.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it can be safely embedded in an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            query_data (dict): Specifies the set of assignments being requested. The value for the key can be: 1. A simple scalar value, which is matched exactly 2. A list of values, which matches any element in the list 3. A dictionary with keys lower_in/lower_ex and upper_in/upper_ex, which matches values in an inclusive/exclusive range
            add_constraints (list): List of constraints dict to apply to the query. The constraint dict should have the following keys: 1. query (dict): Specifies the set of assignments involved in the constraint. The format is same as query_data. 2. operator (str): Constraint operator '=' or '<=' or '>='. 3. constant (int): Constraint RHS constant value."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getAlternativeAssignments', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'addConstraints': add_constraints}, server_override=prediction_url)

    def check_constraints(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Check for any constraints violated by the overrides.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model within an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            query_data (dict): Assignment overrides to the solution."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('checkConstraints', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def predict_with_binary_data(self, deployment_token: str, deployment_id: str, blob: io.TextIOBase) -> Dict:
        """Make predictions for a given blob, e.g. image, audio

        Args:
            deployment_token (str): A token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model in an application or website.
            deployment_id (str): A unique identifier to a deployment created under the project.
            blob (io.TextIOBase): The multipart/form-data of the data."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('predictWithBinaryData', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, files={'blob': blob}, server_override=prediction_url)

    def describe_image(self, deployment_token: str, deployment_id: str, image: io.TextIOBase, categories: list, top_n: int = None) -> Dict:
        """Describe the similarity between an image and a list of categories.

        Args:
            deployment_token (str): Authentication token to access created deployments. This token is only authorized to predict on deployments in the current project, and can be safely embedded in an application or website.
            deployment_id (str): Unique identifier of a deployment created under the project.
            image (io.TextIOBase): Image to describe.
            categories (list): List of candidate categories to compare with the image.
            top_n (int): Return the N most similar categories."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('describeImage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id, 'categories': categories, 'topN': top_n}, files={'image': image}, server_override=prediction_url)

    def transcribe_audio(self, deployment_token: str, deployment_id: str, audio: io.TextIOBase) -> Dict:
        """Transcribe the audio

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to make predictions on deployments in this project, so it can be safely embedded in an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            audio (io.TextIOBase): The audio to transcribe."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('transcribeAudio', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, files={'audio': audio}, server_override=prediction_url)

    def classify_image(self, deployment_token: str, deployment_id: str, image: io.TextIOBase = None, doc_id: str = None) -> Dict:
        """Classify an image.

        Args:
            deployment_token (str): A deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier to a deployment created under the project.
            image (io.TextIOBase): The binary data of the image to classify. One of image or doc_id must be specified.
            doc_id (str): The document ID of the image. One of image or doc_id must be specified."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('classifyImage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id, 'docId': doc_id}, files={'image': image}, server_override=prediction_url)

    def classify_pdf(self, deployment_token: str, deployment_id: str, pdf: io.TextIOBase = None) -> Dict:
        """Returns a classification prediction from a PDF

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model within an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            pdf (io.TextIOBase): (Optional) The pdf to predict on. One of pdf or docId must be specified."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('classifyPDF', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, files={'pdf': pdf}, server_override=prediction_url)

    def get_cluster(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Predicts the cluster for given data.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            query_data (dict): A dictionary where each 'key' represents a column name and its corresponding 'value' represents the value of that column. For Timeseries Clustering, the 'key' should be ITEM_ID, and its value should represent a unique item ID that needs clustering."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getCluster', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def get_objects_from_image(self, deployment_token: str, deployment_id: str, image: io.TextIOBase) -> Dict:
        """Classify an image.

        Args:
            deployment_token (str): A deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier to a deployment created under the project.
            image (io.TextIOBase): The binary data of the image to detect objects from."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('getObjectsFromImage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, files={'image': image}, server_override=prediction_url)

    def score_image(self, deployment_token: str, deployment_id: str, image: io.TextIOBase) -> Dict:
        """Score on image.

        Args:
            deployment_token (str): A deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier to a deployment created under the project.
            image (io.TextIOBase): The binary data of the image to get the score."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('scoreImage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, files={'image': image}, server_override=prediction_url)

    def transfer_style(self, deployment_token: str, deployment_id: str, source_image: io.TextIOBase, style_image: io.TextIOBase) -> io.BytesIO:
        """Change the source image to adopt the visual style from the style image.

        Args:
            deployment_token (str): A token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model in an application or website.
            deployment_id (str): A unique identifier to a deployment created under the project.
            source_image (io.TextIOBase): The source image to apply the makeup.
            style_image (io.TextIOBase): The image that has the style as a reference."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('transferStyle', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, files={'sourceImage': source_image, 'styleImage': style_image}, streamable_response=True, server_override=prediction_url)

    def generate_image(self, deployment_token: str, deployment_id: str, query_data: dict) -> io.BytesIO:
        """Generate an image from text prompt.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model within an application or website.
            deployment_id (str): A unique identifier to a deployment created under the project.
            query_data (dict): Specifies the text prompt. For example, {'prompt': 'a cat'}"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('generateImage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, streamable_response=True, server_override=prediction_url)

    def execute_agent(self, deployment_token: str, deployment_id: str, arguments: list = None, keyword_arguments: dict = None) -> Dict:
        """Executes a deployed AI agent function using the arguments as keyword arguments to the agent execute function.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            arguments (list): Positional arguments to the agent execute function.
            keyword_arguments (dict): A dictionary where each 'key' represents the paramter name and its corresponding 'value' represents the value of that parameter for the agent execute function."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('executeAgent', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'arguments': arguments, 'keywordArguments': keyword_arguments}, server_override=prediction_url)

    def create_batch_prediction(self, deployment_id: str, table_name: str = None, name: str = None, global_prediction_args: Union[dict, BatchPredictionArgs] = None, explanations: bool = False, output_format: str = None, output_location: str = None, database_connector_id: str = None, database_output_config: dict = None, refresh_schedule: str = None, csv_input_prefix: str = None, csv_prediction_prefix: str = None, csv_explanations_prefix: str = None, output_includes_metadata: bool = None, result_input_columns: list = None) -> BatchPrediction:
        """Creates a batch prediction job description for the given deployment.

        Args:
            deployment_id (str): Unique string identifier for the deployment.
            table_name (str): Name of the feature group table to write the results of the batch prediction. Can only be specified if outputLocation and databaseConnectorId are not specified. If tableName is specified, the outputType will be enforced as CSV.
            name (str): Name of the batch prediction job.
            global_prediction_args (BatchPredictionArgs): Batch Prediction args specific to problem type.
            explanations (bool): If true, SHAP explanations will be provided for each prediction, if supported by the use case.
            output_format (str): Format of the batch prediction output (CSV or JSON).
            output_location (str): Location to write the prediction results. Otherwise, results will be stored in Abacus.AI.
            database_connector_id (str): Unique identifier of a Database Connection to write predictions to. Cannot be specified in conjunction with outputLocation.
            database_output_config (dict): Key-value pair of columns/values to write to the database connector. Only available if databaseConnectorId is specified.
            refresh_schedule (str): Cron-style string that describes a schedule in UTC to automatically run the batch prediction.
            csv_input_prefix (str): Prefix to prepend to the input columns, only applies when output format is CSV.
            csv_prediction_prefix (str): Prefix to prepend to the prediction columns, only applies when output format is CSV.
            csv_explanations_prefix (str): Prefix to prepend to the explanation columns, only applies when output format is CSV.
            output_includes_metadata (bool): If true, output will contain columns including prediction start time, batch prediction version, and model version.
            result_input_columns (list): If present, will limit result files or feature groups to only include columns present in this list.

        Returns:
            BatchPrediction: The batch prediction description."""
        return self._call_api('createBatchPrediction', 'POST', query_params={'deploymentId': deployment_id}, body={'tableName': table_name, 'name': name, 'globalPredictionArgs': global_prediction_args, 'explanations': explanations, 'outputFormat': output_format, 'outputLocation': output_location, 'databaseConnectorId': database_connector_id, 'databaseOutputConfig': database_output_config, 'refreshSchedule': refresh_schedule, 'csvInputPrefix': csv_input_prefix, 'csvPredictionPrefix': csv_prediction_prefix, 'csvExplanationsPrefix': csv_explanations_prefix, 'outputIncludesMetadata': output_includes_metadata, 'resultInputColumns': result_input_columns}, parse_type=BatchPrediction)

    def start_batch_prediction(self, batch_prediction_id: str) -> BatchPredictionVersion:
        """Creates a new batch prediction version job for a given batch prediction job description.

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction to create a new version of.

        Returns:
            BatchPredictionVersion: The batch prediction version started by this method call."""
        return self._call_api('startBatchPrediction', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id}, parse_type=BatchPredictionVersion)

    def update_batch_prediction(self, batch_prediction_id: str, deployment_id: str = None, global_prediction_args: Union[dict, BatchPredictionArgs] = None, explanations: bool = None, output_format: str = None, csv_input_prefix: str = None, csv_prediction_prefix: str = None, csv_explanations_prefix: str = None, output_includes_metadata: bool = None, result_input_columns: list = None, name: str = None) -> BatchPrediction:
        """Update a batch prediction job description.

        Args:
            batch_prediction_id (str): Unique identifier of the batch prediction.
            deployment_id (str): Unique identifier of the deployment.
            global_prediction_args (BatchPredictionArgs): Batch Prediction args specific to problem type.
            explanations (bool): If True, SHAP explanations for each prediction will be provided, if supported by the use case.
            output_format (str): If specified, sets the format of the batch prediction output (CSV or JSON).
            csv_input_prefix (str): Prefix to prepend to the input columns, only applies when output format is CSV.
            csv_prediction_prefix (str): Prefix to prepend to the prediction columns, only applies when output format is CSV.
            csv_explanations_prefix (str): Prefix to prepend to the explanation columns, only applies when output format is CSV.
            output_includes_metadata (bool): If True, output will contain columns including prediction start time, batch prediction version, and model version.
            result_input_columns (list): If present, will limit result files or feature groups to only include columns present in this list.
            name (str): If present, will rename the batch prediction.

        Returns:
            BatchPrediction: The batch prediction."""
        return self._call_api('updateBatchPrediction', 'POST', query_params={'deploymentId': deployment_id}, body={'batchPredictionId': batch_prediction_id, 'globalPredictionArgs': global_prediction_args, 'explanations': explanations, 'outputFormat': output_format, 'csvInputPrefix': csv_input_prefix, 'csvPredictionPrefix': csv_prediction_prefix, 'csvExplanationsPrefix': csv_explanations_prefix, 'outputIncludesMetadata': output_includes_metadata, 'resultInputColumns': result_input_columns, 'name': name}, parse_type=BatchPrediction)

    def set_batch_prediction_file_connector_output(self, batch_prediction_id: str, output_format: str = None, output_location: str = None) -> BatchPrediction:
        """Updates the file connector output configuration of the batch prediction

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction.
            output_format (str): The format of the batch prediction output (CSV or JSON). If not specified, the default format will be used.
            output_location (str): The location to write the prediction results. If not specified, results will be stored in Abacus.AI.

        Returns:
            BatchPrediction: The batch prediction description."""
        return self._call_api('setBatchPredictionFileConnectorOutput', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id, 'outputFormat': output_format, 'outputLocation': output_location}, parse_type=BatchPrediction)

    def set_batch_prediction_database_connector_output(self, batch_prediction_id: str, database_connector_id: str = None, database_output_config: dict = None) -> BatchPrediction:
        """Updates the database connector output configuration of the batch prediction

        Args:
            batch_prediction_id (str): Unique string identifier of the batch prediction.
            database_connector_id (str): Unique string identifier of an Database Connection to write predictions to.
            database_output_config (dict): Key-value pair of columns/values to write to the database connector.

        Returns:
            BatchPrediction: Description of the batch prediction."""
        return self._call_api('setBatchPredictionDatabaseConnectorOutput', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id, 'databaseConnectorId': database_connector_id, 'databaseOutputConfig': database_output_config}, parse_type=BatchPrediction)

    def set_batch_prediction_feature_group_output(self, batch_prediction_id: str, table_name: str) -> BatchPrediction:
        """Creates a feature group and sets it as the batch prediction output.

        Args:
            batch_prediction_id (str): Unique string identifier of the batch prediction.
            table_name (str): Name of the feature group table to create.

        Returns:
            BatchPrediction: Batch prediction after the output has been applied."""
        return self._call_api('setBatchPredictionFeatureGroupOutput', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id, 'tableName': table_name}, parse_type=BatchPrediction)

    def set_batch_prediction_output_to_console(self, batch_prediction_id: str) -> BatchPrediction:
        """Sets the batch prediction output to the console, clearing both the file connector and database connector configurations.

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction.

        Returns:
            BatchPrediction: The batch prediction description."""
        return self._call_api('setBatchPredictionOutputToConsole', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id}, parse_type=BatchPrediction)

    def set_batch_prediction_feature_group(self, batch_prediction_id: str, feature_group_type: str, feature_group_id: str = None) -> BatchPrediction:
        """Sets the batch prediction input feature group.

        Args:
            batch_prediction_id (str): Unique identifier of the batch prediction.
            feature_group_type (str): Enum string representing the feature group type to set. The type is based on the use case under which the feature group is being created (e.g. Catalog Attributes for personalized recommendation use case).
            feature_group_id (str): Unique identifier of the feature group to set as input to the batch prediction.

        Returns:
            BatchPrediction: Description of the batch prediction."""
        return self._call_api('setBatchPredictionFeatureGroup', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id, 'featureGroupType': feature_group_type, 'featureGroupId': feature_group_id}, parse_type=BatchPrediction)

    def set_batch_prediction_dataset_remap(self, batch_prediction_id: str, dataset_id_remap: dict) -> BatchPrediction:
        """For the purpose of this batch prediction, will swap out datasets in the training feature groups

        Args:
            batch_prediction_id (str): Unique string identifier of the batch prediction.
            dataset_id_remap (dict): Key/value pairs of dataset ids to be replaced during the batch prediction.

        Returns:
            BatchPrediction: Batch prediction object."""
        return self._call_api('setBatchPredictionDatasetRemap', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id, 'datasetIdRemap': dataset_id_remap}, parse_type=BatchPrediction)

    def delete_batch_prediction(self, batch_prediction_id: str):
        """Deletes a batch prediction and associated data, such as associated monitors.

        Args:
            batch_prediction_id (str): Unique string identifier of the batch prediction."""
        return self._call_api('deleteBatchPrediction', 'DELETE', query_params={'batchPredictionId': batch_prediction_id})

    def add_user_item_interaction(self, streaming_token: str, dataset_id: str, timestamp: int, user_id: str, item_id: list, event_type: str, additional_attributes: dict):
        """Adds a user-item interaction record (data row) to a streaming dataset.

        Args:
            streaming_token (str): The streaming token for authenticating requests to the dataset.
            dataset_id (str): The unique string identifier for the streaming dataset to record data to.
            timestamp (int): The Unix timestamp of the event.
            user_id (str): The unique identifier for the user.
            item_id (list): The unique identifier for the items.
            event_type (str): The event type.
            additional_attributes (dict): Attributes of the user interaction."""
        return self._call_api('addUserItemInteraction', 'POST', query_params={'streamingToken': streaming_token, 'datasetId': dataset_id}, body={'timestamp': timestamp, 'userId': user_id, 'itemId': item_id, 'eventType': event_type, 'additionalAttributes': additional_attributes})

    def upsert_user_attributes(self, streaming_token: str, dataset_id: str, user_id: str, user_attributes: dict):
        """Adds a user attribute record (data row) to a streaming dataset.

        Either the streaming dataset ID or the project ID is required.


        Args:
            streaming_token (str): The streaming token for authenticating requests to the dataset.
            dataset_id (str): The unique string identifier for the streaming dataset to record data to.
            user_id (str): The unique identifier for the user.
            user_attributes (dict): Attributes of the user interaction."""
        return self._call_api('upsertUserAttributes', 'POST', query_params={'streamingToken': streaming_token, 'datasetId': dataset_id}, body={'userId': user_id, 'userAttributes': user_attributes})

    def upsert_item_attributes(self, streaming_token: str, dataset_id: str, item_id: str, item_attributes: dict):
        """Adds an item attributes record (data row) to a streaming dataset.

        Either the streaming dataset ID or the project ID is required.


        Args:
            streaming_token (str): The streaming token for authenticating requests to the dataset.
            dataset_id (str): The unique string identifier for the streaming dataset to record data to.
            item_id (str): The unique identifier for the item.
            item_attributes (dict): Attributes of the item interaction."""
        return self._call_api('upsertItemAttributes', 'POST', query_params={'streamingToken': streaming_token, 'datasetId': dataset_id}, body={'itemId': item_id, 'itemAttributes': item_attributes})

    def add_multiple_user_item_interactions(self, streaming_token: str, dataset_id: str, interactions: list):
        """Adds a user-item interaction record (data row) to a streaming dataset.

        Args:
            streaming_token (str): The streaming token for authenticating requests to the dataset.
            dataset_id (str): The unique string identifier of the streaming dataset to record data to.
            interactions (list): List of interactions, each interaction of format {'userId': userId, 'timestamp': timestamp, 'itemId': itemId, 'eventType': eventType, 'additionalAttributes': {'attribute1': 'abc', 'attribute2': 123}}."""
        return self._call_api('addMultipleUserItemInteractions', 'POST', query_params={'streamingToken': streaming_token, 'datasetId': dataset_id}, body={'interactions': interactions})

    def upsert_multiple_user_attributes(self, streaming_token: str, dataset_id: str, upserts: list):
        """Adds multiple user attributes records (data rows) to a streaming dataset.

        Args:
            streaming_token (str): The streaming token for authenticating requests to the dataset.
            dataset_id (str): A unique string identifier for the streaming dataset to record data to.
            upserts (list): List of upserts, each upsert of format {'userId': userId, 'userAttributes': {'attribute1': 'abc', 'attribute2': 123}}."""
        return self._call_api('upsertMultipleUserAttributes', 'POST', query_params={'streamingToken': streaming_token, 'datasetId': dataset_id}, body={'upserts': upserts})

    def upsert_multiple_item_attributes(self, streaming_token: str, dataset_id: str, upserts: list):
        """Adds multiple item attributes records (data rows) to a streaming dataset.

        Args:
            streaming_token (str): The streaming token for authenticating requests to the dataset.
            dataset_id (str): A unique string identifier for the streaming dataset to record data to.
            upserts (list): A list of upserts, each upsert of format {'itemId': itemId, 'itemAttributes': {'attribute1': 'abc', 'attribute2': 123}}."""
        return self._call_api('upsertMultipleItemAttributes', 'POST', query_params={'streamingToken': streaming_token, 'datasetId': dataset_id}, body={'upserts': upserts})

    def upsert_item_embeddings(self, streaming_token: str, model_id: str, item_id: str, vector: list, catalog_id: str = None):
        """Upserts an embedding vector for an item id for a model_id.

        Args:
            streaming_token (str): The streaming token for authenticating requests to the model.
            model_id (str): A unique string identifier for the model to upsert item embeddings to.
            item_id (str): The item id for which its embeddings will be upserted.
            vector (list): The embedding vector.
            catalog_id (str): The name of the catalog in the model to update."""
        prediction_url = self._get_streaming_endpoint(
            streaming_token, model_id=model_id)
        return self._call_api('upsertItemEmbeddings', 'POST', query_params={'streamingToken': streaming_token}, body={'modelId': model_id, 'itemId': item_id, 'vector': vector, 'catalogId': catalog_id}, server_override=prediction_url)

    def delete_item_embeddings(self, streaming_token: str, model_id: str, item_ids: list, catalog_id: str = None):
        """Deletes KNN embeddings for a list of item IDs for a given model ID.

        Args:
            streaming_token (str): The streaming token for authenticating requests to the model.
            model_id (str): A unique string identifier for the model from which to delete item embeddings.
            item_ids (list): A list of item IDs whose embeddings will be deleted.
            catalog_id (str): An optional name to specify which catalog in a model to update."""
        prediction_url = self._get_streaming_endpoint(
            streaming_token, model_id=model_id)
        return self._call_api('deleteItemEmbeddings', 'POST', query_params={'streamingToken': streaming_token}, body={'modelId': model_id, 'itemIds': item_ids, 'catalogId': catalog_id}, server_override=prediction_url)

    def upsert_multiple_item_embeddings(self, streaming_token: str, model_id: str, upserts: list, catalog_id: str = None):
        """Upserts a knn embedding for multiple item ids for a model_id.

        Args:
            streaming_token (str): The streaming token for authenticating requests to the model.
            model_id (str): The unique string identifier of the model to upsert item embeddings to.
            upserts (list): A list of dictionaries of the form {'itemId': ..., 'vector': [...]} for each upsert.
            catalog_id (str): Name of the catalog in the model to update."""
        prediction_url = self._get_streaming_endpoint(
            streaming_token, model_id=model_id)
        return self._call_api('upsertMultipleItemEmbeddings', 'POST', query_params={'streamingToken': streaming_token}, body={'modelId': model_id, 'upserts': upserts, 'catalogId': catalog_id}, server_override=prediction_url)

    def upsert_data(self, feature_group_id: str, streaming_token: str, data: dict):
        """Update new data into the feature group for a given lookup key record ID if the record ID is found; otherwise, insert new data into the feature group.

        Args:
            feature_group_id (str): A unique string identifier of the streaming feature group to record data to.
            streaming_token (str): The streaming token for authenticating requests.
            data (dict): The data to record, in JSON format."""
        prediction_url = self._get_streaming_endpoint(
            streaming_token, feature_group_id=feature_group_id)
        return self._call_api('upsertData', 'POST', query_params={'streamingToken': streaming_token}, body={'featureGroupId': feature_group_id, 'data': data}, server_override=prediction_url)

    def append_data(self, feature_group_id: str, streaming_token: str, data: dict):
        """Appends new data into the feature group for a given lookup key recordId.

        Args:
            feature_group_id (str): Unique string identifier for the streaming feature group to record data to.
            streaming_token (str): The streaming token for authenticating requests.
            data (dict): The data to record as a JSON object."""
        prediction_url = self._get_streaming_endpoint(
            streaming_token, feature_group_id=feature_group_id)
        return self._call_api('appendData', 'POST', query_params={'streamingToken': streaming_token}, body={'featureGroupId': feature_group_id, 'data': data}, server_override=prediction_url)

    def upsert_multiple_data(self, feature_group_id: str, streaming_token: str, data: list):
        """Update new data into the feature group for a given lookup key recordId if the recordId is found; otherwise, insert new data into the feature group.

        Args:
            feature_group_id (str): Unique string identifier for the streaming feature group to record data to.
            streaming_token (str): The streaming token for authenticating requests.
            data (list): The data to record, as a list of JSON objects."""
        prediction_url = self._get_streaming_endpoint(
            streaming_token, feature_group_id=feature_group_id)
        return self._call_api('upsertMultipleData', 'POST', query_params={'streamingToken': streaming_token}, body={'featureGroupId': feature_group_id, 'data': data}, server_override=prediction_url)

    def append_multiple_data(self, feature_group_id: str, streaming_token: str, data: list):
        """Appends new data into the feature group for a given lookup key recordId.

        Args:
            feature_group_id (str): Unique string identifier of the streaming feature group to record data to.
            streaming_token (str): Streaming token for authenticating requests.
            data (list): Data to record, as a list of JSON objects."""
        prediction_url = self._get_streaming_endpoint(
            streaming_token, feature_group_id=feature_group_id)
        return self._call_api('appendMultipleData', 'POST', query_params={'streamingToken': streaming_token}, body={'featureGroupId': feature_group_id, 'data': data}, server_override=prediction_url)

    def create_python_function(self, name: str, source_code: str = None, function_name: str = None, function_variable_mappings: list = None, package_requirements: list = None, function_type: str = 'FEATURE_GROUP') -> PythonFunction:
        """Creates a custom Python function that is reusable.

        Args:
            name (str): The name to identify the Python function.
            source_code (str): Contents of a valid Python source code file. The source code should contain the transform feature group functions. A list of allowed imports and system libraries for each language is specified in the user functions documentation section.
            function_name (str): The name of the Python function.
            function_variable_mappings (list): List of Python function arguments.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            function_type (str): Type of Python function to create.

        Returns:
            PythonFunction: The Python function that can be used (e.g. for feature group transform)."""
        return self._call_api('createPythonFunction', 'POST', query_params={}, body={'name': name, 'sourceCode': source_code, 'functionName': function_name, 'functionVariableMappings': function_variable_mappings, 'packageRequirements': package_requirements, 'functionType': function_type}, parse_type=PythonFunction)

    def update_python_function(self, name: str, source_code: str = None, function_name: str = None, function_variable_mappings: list = None, package_requirements: list = None) -> PythonFunction:
        """Update custom python function with user inputs for the given python function.

        Args:
            name (str): The name to identify the Python function.
            source_code (str): Contents of a valid Python source code file. The source code should contain the transform feature group functions. A list of allowed imports and system libraries for each language is specified in the user functions documentation section.
            function_name (str): The name of the Python function within `source_code`.
            function_variable_mappings (list): List of arguments required by `function_name`.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].

        Returns:
            PythonFunction: The Python function object."""
        return self._call_api('updatePythonFunction', 'PATCH', query_params={}, body={'name': name, 'sourceCode': source_code, 'functionName': function_name, 'functionVariableMappings': function_variable_mappings, 'packageRequirements': package_requirements}, parse_type=PythonFunction)

    def delete_python_function(self, name: str):
        """Removes an existing Python function.

        Args:
            name (str): The name to identify the Python function."""
        return self._call_api('deletePythonFunction', 'DELETE', query_params={'name': name})

    def create_pipeline(self, pipeline_name: str, project_id: str = None, cron: str = None, is_prod: bool = None) -> Pipeline:
        """Creates a pipeline for executing multiple steps.

        Args:
            pipeline_name (str): The name of the pipeline, which should be unique to the organization.
            project_id (str): A unique string identifier for the pipeline.
            cron (str): A cron-like string specifying the frequency of pipeline reruns.
            is_prod (bool): Whether the pipeline is a production pipeline or not.

        Returns:
            Pipeline: An object that describes a Pipeline."""
        return self._call_api('createPipeline', 'POST', query_params={}, body={'pipelineName': pipeline_name, 'projectId': project_id, 'cron': cron, 'isProd': is_prod}, parse_type=Pipeline)

    def describe_pipeline(self, pipeline_id: str) -> Pipeline:
        """Describes a given pipeline.

        Args:
            pipeline_id (str): The ID of the pipeline to describe.

        Returns:
            Pipeline: An object describing a Pipeline"""
        return self._call_api('describePipeline', 'POST', query_params={}, body={'pipelineId': pipeline_id}, parse_type=Pipeline)

    def describe_pipeline_by_name(self, pipeline_name: str) -> Pipeline:
        """Describes a given pipeline.

        Args:
            pipeline_name (str): The name of the pipeline to describe.

        Returns:
            Pipeline: An object describing a Pipeline"""
        return self._call_api('describePipelineByName', 'POST', query_params={}, body={'pipelineName': pipeline_name}, parse_type=Pipeline)

    def update_pipeline(self, pipeline_id: str, project_id: str = None, pipeline_variable_mappings: list = None, cron: str = None, is_prod: bool = None) -> Pipeline:
        """Updates a pipeline for executing multiple steps.

        Args:
            pipeline_id (str): The ID of the pipeline to update.
            project_id (str): A unique string identifier for the pipeline.
            pipeline_variable_mappings (list): List of Python function arguments for the pipeline.
            cron (str): A cron-like string specifying the frequency of the scheduled pipeline runs.
            is_prod (bool): Whether the pipeline is a production pipeline or not.

        Returns:
            Pipeline: An object that describes a Pipeline."""
        return self._call_api('updatePipeline', 'POST', query_params={}, body={'pipelineId': pipeline_id, 'projectId': project_id, 'pipelineVariableMappings': pipeline_variable_mappings, 'cron': cron, 'isProd': is_prod}, parse_type=Pipeline)

    def delete_pipeline(self, pipeline_id: str):
        """Deletes a pipeline.

        Args:
            pipeline_id (str): The ID of the pipeline to delete."""
        return self._call_api('deletePipeline', 'DELETE', query_params={'pipelineId': pipeline_id})

    def list_pipeline_versions(self, pipeline_id: str, limit: int = 200) -> List[PipelineVersion]:
        """Lists the pipeline versions for a specified pipeline

        Args:
            pipeline_id (str): The ID of the pipeline to list versions for.
            limit (int): The maximum number of pipeline versions to return.

        Returns:
            list[PipelineVersion]: A list of pipeline versions."""
        return self._call_api('listPipelineVersions', 'POST', query_params={}, body={'pipelineId': pipeline_id, 'limit': limit}, parse_type=PipelineVersion)

    def run_pipeline(self, pipeline_id: str, pipeline_variable_mappings: list = None) -> PipelineVersion:
        """Runs a specified pipeline with the arguments provided.

        Args:
            pipeline_id (str): The ID of the pipeline to run.
            pipeline_variable_mappings (list): List of Python function arguments for the pipeline.

        Returns:
            PipelineVersion: The object describing the pipeline"""
        return self._call_api('runPipeline', 'POST', query_params={}, body={'pipelineId': pipeline_id, 'pipelineVariableMappings': pipeline_variable_mappings}, parse_type=PipelineVersion)

    def create_pipeline_step(self, pipeline_id: str, step_name: str, function_name: str = None, source_code: str = None, step_input_mappings: list = None, output_variable_mappings: list = None, step_dependencies: list = None, package_requirements: list = None, cpu_size: str = None, memory: int = None) -> Pipeline:
        """Creates a step in a given pipeline.

        Args:
            pipeline_id (str): The ID of the pipeline to run.
            step_name (str): The name of the step.
            function_name (str): The name of the Python function.
            source_code (str): Contents of a valid Python source code file. The source code should contain the transform feature group functions. A list of allowed imports and system libraries for each language is specified in the user functions documentation section.
            step_input_mappings (list): List of Python function arguments.
            output_variable_mappings (list): List of Python function ouputs.
            step_dependencies (list): List of step names this step depends on.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            cpu_size (str): Size of the CPU for the step function.
            memory (int): Memory (in GB) for the step function.

        Returns:
            Pipeline: Object describing the pipeline."""
        return self._call_api('createPipelineStep', 'POST', query_params={}, body={'pipelineId': pipeline_id, 'stepName': step_name, 'functionName': function_name, 'sourceCode': source_code, 'stepInputMappings': step_input_mappings, 'outputVariableMappings': output_variable_mappings, 'stepDependencies': step_dependencies, 'packageRequirements': package_requirements, 'cpuSize': cpu_size, 'memory': memory}, parse_type=Pipeline)

    def delete_pipeline_step(self, pipeline_step_id: str):
        """Deletes a step from a pipeline.

        Args:
            pipeline_step_id (str): The ID of the pipeline step."""
        return self._call_api('deletePipelineStep', 'DELETE', query_params={'pipelineStepId': pipeline_step_id})

    def update_pipeline_step(self, pipeline_step_id: str, function_name: str = None, source_code: str = None, step_input_mappings: list = None, output_variable_mappings: list = None, step_dependencies: list = None, package_requirements: list = None, cpu_size: str = None, memory: int = None) -> PipelineStep:
        """Creates a step in a given pipeline.

        Args:
            pipeline_step_id (str): The ID of the pipeline_step to update.
            function_name (str): The name of the Python function.
            source_code (str): Contents of a valid Python source code file. The source code should contain the transform feature group functions. A list of allowed imports and system libraries for each language is specified in the user functions documentation section.
            step_input_mappings (list): List of Python function arguments.
            output_variable_mappings (list): List of Python function ouputs.
            step_dependencies (list): List of step names this step depends on.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            cpu_size (str): Size of the CPU for the step function.
            memory (int): Memory (in GB) for the step function.

        Returns:
            PipelineStep: Object describing the pipeline."""
        return self._call_api('updatePipelineStep', 'POST', query_params={}, body={'pipelineStepId': pipeline_step_id, 'functionName': function_name, 'sourceCode': source_code, 'stepInputMappings': step_input_mappings, 'outputVariableMappings': output_variable_mappings, 'stepDependencies': step_dependencies, 'packageRequirements': package_requirements, 'cpuSize': cpu_size, 'memory': memory}, parse_type=PipelineStep)

    def create_graph_dashboard(self, project_id: str, name: str, python_function_ids: list = None) -> GraphDashboard:
        """Create a plot dashboard given selected python plots

        Args:
            project_id (str): A unique string identifier for the plot dashboard.
            name (str): The name of the dashboard.
            python_function_ids (list): A list of unique string identifiers for the python functions to be used in the graph dashboard.

        Returns:
            GraphDashboard: An object describing the graph dashboard."""
        return self._call_api('createGraphDashboard', 'POST', query_params={}, body={'projectId': project_id, 'name': name, 'pythonFunctionIds': python_function_ids}, parse_type=GraphDashboard)

    def delete_graph_dashboard(self, graph_dashboard_id: str):
        """Deletes a graph dashboard

        Args:
            graph_dashboard_id (str): Unique string identifier for the graph dashboard to be deleted."""
        return self._call_api('deleteGraphDashboard', 'DELETE', query_params={'graphDashboardId': graph_dashboard_id})

    def update_graph_dashboard(self, graph_dashboard_id: str, name: str = None, python_function_ids: list = None) -> GraphDashboard:
        """Updates a graph dashboard

        Args:
            graph_dashboard_id (str): Unique string identifier for the graph dashboard to update.
            name (str): Name of the dashboard.
            python_function_ids (list): List of unique string identifiers for the Python functions to be used in the graph dashboard.

        Returns:
            GraphDashboard: An object describing the graph dashboard."""
        return self._call_api('updateGraphDashboard', 'POST', query_params={}, body={'graphDashboardId': graph_dashboard_id, 'name': name, 'pythonFunctionIds': python_function_ids}, parse_type=GraphDashboard)

    def add_graph_to_dashboard(self, python_function_id: str, graph_dashboard_id: str, function_variable_mappings: dict = None, name: str = None) -> GraphDashboard:
        """Add a python plot function to a dashboard

        Args:
            python_function_id (str): Unique string identifier for the Python function.
            graph_dashboard_id (str): Unique string identifier for the graph dashboard to update.
            function_variable_mappings (dict): List of arguments to be supplied to the function as parameters, in the format [{'name': 'function_argument', 'variable_type': 'FEATURE_GROUP', 'value': 'name_of_feature_group'}].
            name (str): Name of the added python plot

        Returns:
            GraphDashboard: An object describing the graph dashboard."""
        return self._call_api('addGraphToDashboard', 'POST', query_params={}, body={'pythonFunctionId': python_function_id, 'graphDashboardId': graph_dashboard_id, 'functionVariableMappings': function_variable_mappings, 'name': name}, parse_type=GraphDashboard)

    def update_graph_to_dashboard(self, graph_reference_id: str, function_variable_mappings: list = None, name: str = None) -> GraphDashboard:
        """Update a python plot function to a dashboard

        Args:
            graph_reference_id (str): A unique string identifier for the graph reference.
            function_variable_mappings (list): A list of arguments to be supplied to the Python function as parameters in the format [{'name': 'function_argument', 'variable_type': 'FEATURE_GROUP', 'value': 'name_of_feature_group'}].
            name (str): The updated name for the graph

        Returns:
            GraphDashboard: An object describing the graph dashboard."""
        return self._call_api('updateGraphToDashboard', 'PATCH', query_params={}, body={'graphReferenceId': graph_reference_id, 'functionVariableMappings': function_variable_mappings, 'name': name}, parse_type=GraphDashboard)

    def create_algorithm(self, name: str, problem_type: str, source_code: str = None, training_data_parameter_names_mapping: dict = None, training_config_parameter_name: str = None, train_function_name: str = None, predict_function_name: str = None, predict_many_function_name: str = None, initialize_function_name: str = None, config_options: dict = None, is_default_enabled: bool = False, project_id: str = None, use_gpu: bool = False, package_requirements: list = None) -> Algorithm:
        """Creates a custom algorithm that is re-usable for model training.

        Args:
            name (str): The name to identify the algorithm; only uppercase letters, numbers, and underscores are allowed.
            problem_type (str): The type of problem this algorithm will work on.
            source_code (str): Contents of a valid Python source code file. The source code should contain the train/predict/predict_many/initialize functions. A list of allowed import and system libraries for each language is specified in the user functions documentation section.
            training_data_parameter_names_mapping (dict): The mapping from feature group types to training data parameter names in the train function.
            training_config_parameter_name (str): The train config parameter name in the train function.
            train_function_name (str): Name of the function found in the source code that will be executed to train the model. It is not executed when this function is run.
            predict_function_name (str): Name of the function found in the source code that will be executed to run predictions through the model. It is not executed when this function is run.
            predict_many_function_name (str): Name of the function found in the source code that will be executed for batch prediction of the model. It is not executed when this function is run.
            initialize_function_name (str): Name of the function found in the source code to initialize the trained model before using it to make predictions using the model.
            config_options (dict): Map dataset types and configs to train function parameter names.
            is_default_enabled (bool): Whether to train with the algorithm by default.
            project_id (str): The unique version ID of the project.
            use_gpu (bool): Whether this algorithm needs to run on GPU.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].

        Returns:
            Algorithm: The new custom model that can be used for training."""
        return self._call_api('createAlgorithm', 'POST', query_params={}, body={'name': name, 'problemType': problem_type, 'sourceCode': source_code, 'trainingDataParameterNamesMapping': training_data_parameter_names_mapping, 'trainingConfigParameterName': training_config_parameter_name, 'trainFunctionName': train_function_name, 'predictFunctionName': predict_function_name, 'predictManyFunctionName': predict_many_function_name, 'initializeFunctionName': initialize_function_name, 'configOptions': config_options, 'isDefaultEnabled': is_default_enabled, 'projectId': project_id, 'useGpu': use_gpu, 'packageRequirements': package_requirements}, parse_type=Algorithm)

    def delete_algorithm(self, algorithm: str):
        """Deletes the specified customer algorithm.

        Args:
            algorithm (str): The name of the algorithm to delete."""
        return self._call_api('deleteAlgorithm', 'DELETE', query_params={'algorithm': algorithm})

    def update_algorithm(self, algorithm: str, source_code: str = None, training_data_parameter_names_mapping: dict = None, training_config_parameter_name: str = None, train_function_name: str = None, predict_function_name: str = None, predict_many_function_name: str = None, initialize_function_name: str = None, config_options: dict = None, is_default_enabled: bool = None, use_gpu: bool = None, package_requirements: list = None) -> Algorithm:
        """Update a custom algorithm for the given algorithm name. If source code is provided, all function names for the source code must also be provided.

        Args:
            algorithm (str): The name to identify the algorithm. Only uppercase letters, numbers, and underscores are allowed.
            source_code (str): Contents of a valid Python source code file. The source code should contain the train/predict/predict_many/initialize functions. A list of allowed imports and system libraries for each language is specified in the user functions documentation section.
            training_data_parameter_names_mapping (dict): The mapping from feature group types to training data parameter names in the train function.
            training_config_parameter_name (str): The train config parameter name in the train function.
            train_function_name (str): Name of the function found in the source code that will be executed to train the model. It is not executed when this function is run.
            predict_function_name (str): Name of the function found in the source code that will be executed to run predictions through the model. It is not executed when this function is run.
            predict_many_function_name (str): Name of the function found in the source code that will be executed for batch prediction of the model. It is not executed when this function is run.
            initialize_function_name (str): Name of the function found in the source code to initialize the trained model before using it to make predictions using the model.
            config_options (dict): Map dataset types and configs to train function parameter names.
            is_default_enabled (bool): Whether to train with the algorithm by default.
            use_gpu (bool): Whether this algorithm needs to run on GPU.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].

        Returns:
            Algorithm: The new custom model can be used for training."""
        return self._call_api('updateAlgorithm', 'PATCH', query_params={}, body={'algorithm': algorithm, 'sourceCode': source_code, 'trainingDataParameterNamesMapping': training_data_parameter_names_mapping, 'trainingConfigParameterName': training_config_parameter_name, 'trainFunctionName': train_function_name, 'predictFunctionName': predict_function_name, 'predictManyFunctionName': predict_many_function_name, 'initializeFunctionName': initialize_function_name, 'configOptions': config_options, 'isDefaultEnabled': is_default_enabled, 'useGpu': use_gpu, 'packageRequirements': package_requirements}, parse_type=Algorithm)

    def list_builtin_algorithms(self, project_id: str, feature_group_ids: list, training_config: dict = None) -> List[Algorithm]:
        """Return list of built-in algorithms based on given input.

        Args:
            project_id (str): Unique string identifier associated with the project.
            feature_group_ids (list): List of feature group identifiers applied to the algorithms.
            training_config (dict): The training configuration key/value pairs used to train with the algorithm.

        Returns:
            list[Algorithm]: List of applicable builtin algorithms."""
        return self._call_api('listBuiltinAlgorithms', 'POST', query_params={}, body={'projectId': project_id, 'featureGroupIds': feature_group_ids, 'trainingConfig': training_config}, parse_type=Algorithm)

    def create_custom_loss_function_with_source_code(self, name: str, loss_function_type: str, loss_function_name: str, loss_function_source_code: str) -> CustomLossFunction:
        """Registers a new custom loss function which can be used as an objective function during model training.

        Args:
            name (str): A name for the loss, unique per organization. Must be 50 characters or fewer, and can contain only underscores, numbers, and uppercase alphabets.
            loss_function_type (str): The category of problems that this loss would be applicable to, e.g. REGRESSION_DL_TF, CLASSIFICATION_DL_TF, etc.
            loss_function_name (str): The name of the function whose full source code is passed in loss_function_source_code.
            loss_function_source_code (str): Python source code string of the function.

        Returns:
            CustomLossFunction: A description of the registered custom loss function."""
        return self._call_api('createCustomLossFunctionWithSourceCode', 'POST', query_params={}, body={'name': name, 'lossFunctionType': loss_function_type, 'lossFunctionName': loss_function_name, 'lossFunctionSourceCode': loss_function_source_code}, parse_type=CustomLossFunction)

    def update_custom_loss_function_with_source_code(self, name: str, loss_function_name: str, loss_function_source_code: str) -> CustomLossFunction:
        """Updates a previously registered custom loss function with a new function implementation.

        Args:
            name (str): Name of the registered custom loss.
            loss_function_name (str): Name of the function whose full source code is passed in loss_function_source_code.
            loss_function_source_code (str): Python source code string of the function.

        Returns:
            CustomLossFunction: A description of the updated custom loss function."""
        return self._call_api('updateCustomLossFunctionWithSourceCode', 'PATCH', query_params={}, body={'name': name, 'lossFunctionName': loss_function_name, 'lossFunctionSourceCode': loss_function_source_code}, parse_type=CustomLossFunction)

    def delete_custom_loss_function(self, name: str):
        """Deletes a previously registered custom loss function.

        Args:
            name (str): The name of the custom loss function to be deleted."""
        return self._call_api('deleteCustomLossFunction', 'DELETE', query_params={'name': name})

    def create_custom_metric(self, name: str, problem_type: str, custom_metric_function_name: str = None, source_code: str = None) -> CustomMetric:
        """Registers a new custom metric which can be used as an evaluation metric for the trained model.

        Args:
            name (str): A unique name for the metric, with a limit of 50 characters. Only underscores, numbers, and uppercase alphabets are allowed.
            problem_type (str): The problem type that this metric would be applicable to, e.g. REGRESSION, FORECASTING, etc.
            custom_metric_function_name (str): The name of the function whose full source code is passed in source_code.
            source_code (str): The full source code of the custom metric function. This is required if custom_metric_function_name is passed.

        Returns:
            CustomMetric: The newly created custom metric."""
        return self._call_api('createCustomMetric', 'POST', query_params={}, body={'name': name, 'problemType': problem_type, 'customMetricFunctionName': custom_metric_function_name, 'sourceCode': source_code}, parse_type=CustomMetric)

    def update_custom_metric(self, name: str, custom_metric_function_name: str, source_code: str) -> CustomMetric:
        """Updates a previously registered custom metric with a new function implementation.

        Args:
            name (str): Name of the registered custom metric.
            custom_metric_function_name (str): Name of the function whose full source code is passed in `source_code`.
            source_code (str): Python source code string of the function.

        Returns:
            CustomMetric: A description of the updated custom metric."""
        return self._call_api('updateCustomMetric', 'PATCH', query_params={}, body={'name': name, 'customMetricFunctionName': custom_metric_function_name, 'sourceCode': source_code}, parse_type=CustomMetric)

    def delete_custom_metric(self, name: str):
        """Deletes a previously registered custom metric.

        Args:
            name (str): The name of the custom metric to be deleted."""
        return self._call_api('deleteCustomMetric', 'DELETE', query_params={'name': name})

    def create_module(self, name: str, source_code: str = None) -> Module:
        """Creates a module that's re-usable in customer's code, e.g. python function, bring your own algorithm and etc.

        Args:
            name (str): The name to identify the module, only lower case letters and underscore allowed.
            source_code (str): Contents of a valid python source code file.

        Returns:
            Module: The new module"""
        return self._call_api('createModule', 'POST', query_params={}, body={'name': name, 'sourceCode': source_code}, parse_type=Module)

    def delete_module(self, name: str):
        """Deletes the specified customer module.

        Args:
            name (str): The name of the custom module to delete."""
        return self._call_api('deleteModule', 'DELETE', query_params={'name': name})

    def update_module(self, name: str, source_code: str = None) -> Module:
        """Update the module.

        Args:
            name (str): The name to identify the module.
            source_code (str): Contents of a valid python source code file.

        Returns:
            Module: The updated module."""
        return self._call_api('updateModule', 'PATCH', query_params={}, body={'name': name, 'sourceCode': source_code}, parse_type=Module)

    def get_notebook_cell_completion(self, previous_cells: list, completion_type: str = None) -> List[NotebookCompletion]:
        """Calls an autocomplete LLM with the input 'previousCells' which is all the previous context of a notebook in the format [{'type':'code/markdown', 'content':'cell text'}].

        Args:
            previous_cells (list): A list of cells from Notebook for OpenAI to autocomplete.
            completion_type (str): Specify the type based on which the completion is done.

        Returns:
            list[NotebookCompletion]: A list of objects, each containing the type and contents of the new cell to be inserted."""
        return self._call_api('getNotebookCellCompletion', 'POST', query_params={}, body={'previousCells': previous_cells, 'completionType': completion_type}, parse_type=NotebookCompletion)

    def set_natural_language_explanation(self, short_explanation: str, long_explanation: str, feature_group_id: str = None, feature_group_version: str = None, model_id: str = None):
        """Saves the natural language explanation of an artifact with given ID. The artifact can be - Feature Group or Feature Group Version

        Args:
            short_explanation (str): succinct explanation of the artifact with given ID
            long_explanation (str): verbose explanation of the artifact with given ID
            feature_group_id (str): A unique string identifier associated with the Feature Group.
            feature_group_version (str): A unique string identifier associated with the Feature Group Version.
            model_id (str): A unique string identifier associated with the Model."""
        return self._call_api('setNaturalLanguageExplanation', 'POST', query_params={}, body={'shortExplanation': short_explanation, 'longExplanation': long_explanation, 'featureGroupId': feature_group_id, 'featureGroupVersion': feature_group_version, 'modelId': model_id}, timeout=300)

    def create_chat_session(self, project_id: str = None) -> ChatSession:
        """Creates a chat session with Abacus Chat.

        Args:
            project_id (str): The project this chat session belongs to

        Returns:
            ChatSession: The chat session with Abacus Chat"""
        return self._call_api('createChatSession', 'POST', query_params={}, body={'projectId': project_id}, parse_type=ChatSession)

    def create_deployment_conversation(self, deployment_id: str, name: str) -> DeploymentConversation:
        """Creates a deployment conversation.

        Args:
            deployment_id (str): The deployment this conversation belongs to.
            name (str): The name of the conversation.

        Returns:
            DeploymentConversation: The deployment conversation."""
        return self._call_api('createDeploymentConversation', 'POST', query_params={'deploymentId': deployment_id}, body={'name': name}, parse_type=DeploymentConversation)

    def delete_deployment_conversation(self, deployment_conversation_id: str):
        """Delete a Deployment Conversation.

        Args:
            deployment_conversation_id (str): A unique string identifier associated with the deployment conversation."""
        return self._call_api('deleteDeploymentConversation', 'DELETE', query_params={'deploymentConversationId': deployment_conversation_id})

    def set_deployment_conversation_feedback(self, deployment_conversation_id: str, message_index: int, is_useful: bool = None, is_not_useful: bool = None, feedback: str = None):
        """Sets a deployment conversation message as useful or not useful

        Args:
            deployment_conversation_id (str): A unique string identifier associated with the deployment conversation.
            message_index (int): The index of the deployment conversation message
            is_useful (bool): If the message is useful. If true, the message is useful. If false, clear the useful flag.
            is_not_useful (bool): If the message is not useful. If true, the message is not useful. If set to false, clear the useful flag.
            feedback (str): Optional feedback on why the message is useful or not useful"""
        return self._call_api('setDeploymentConversationFeedback', 'POST', query_params={}, body={'deploymentConversationId': deployment_conversation_id, 'messageIndex': message_index, 'isUseful': is_useful, 'isNotUseful': is_not_useful, 'feedback': feedback})

    def rename_deployment_conversation(self, deployment_conversation_id: str, name: str):
        """Rename a Deployment Conversation.

        Args:
            deployment_conversation_id (str): A unique string identifier associated with the deployment conversation.
            name (str): The new name of the conversation."""
        return self._call_api('renameDeploymentConversation', 'POST', query_params={}, body={'deploymentConversationId': deployment_conversation_id, 'name': name})

    def create_agent(self, project_id: str, function_source_code: str, agent_function_name: str, name: str = None, memory: int = None, package_requirements: list = None, description: str = None) -> Model:
        """Creates a new AI agent.

        Args:
            project_id (str): The unique ID associated with the project.
            function_source_code (str): The contents of a valid Python source code file. The source code should contain a function named `agentFunctionName`. A list of allowed import and system libraries for each language is specified in the user functions documentation section.
            agent_function_name (str): The name of the function found in the source code that will be executed when the agent is deployed.
            name (str): The name you want your agent to have, defaults to "<Project Name> Agent".
            memory (int): The memory allocation (in GB) for the agent.
            package_requirements (list): A list of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            description (str): A description of the agent, including its purpose and instructions.

        Returns:
            Model: The new agent"""
        return self._call_api('createAgent', 'POST', query_params={}, body={'projectId': project_id, 'functionSourceCode': function_source_code, 'agentFunctionName': agent_function_name, 'name': name, 'memory': memory, 'packageRequirements': package_requirements, 'description': description}, parse_type=Model)

    def update_agent(self, model_id: str, function_source_code: str = None, agent_function_name: str = None, memory: int = None, package_requirements: list = None, description: str = None) -> Model:
        """Updates an existing AI Agent using user-provided Python code. A new version of the agent will be created and published.

        Args:
            model_id (str): The unique ID associated with the AI Agent to be changed.
            function_source_code (str): Contents of a valid Python source code file. The source code should contain the functions named `agentFunctionName`. A list of allowed import and system libraries for each language is specified in the user functions documentation section.
            agent_function_name (str): Name of the function found in the source code that will be executed to the agent when it is deployed.
            memory (int): Memory (in GB) for the agent.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            description (str): A description of the agent, including its purpose and instructions.

        Returns:
            Model: The updated agent"""
        return self._call_api('updateAgent', 'POST', query_params={}, body={'modelId': model_id, 'functionSourceCode': function_source_code, 'agentFunctionName': agent_function_name, 'memory': memory, 'packageRequirements': package_requirements, 'description': description}, parse_type=Model)

    def evaluate_prompt(self, prompt: str, system_message: str = None, llm_name: str = None, max_tokens: int = None) -> LlmResponse:
        """Generate response to the prompt using the specified model.

        Args:
            prompt (str): Prompt to use for generation.
            system_message (str): System message for models that support it.
            llm_name (str): Name of the underlying LLM to be used for generation. Should be one of 'gpt-4' or 'gpt-3.5-turbo'. Default is auto selection.
            max_tokens (int): Maximum number of tokens to generate. If set, the model will just stop generating after this token limit is reached.

        Returns:
            LlmResponse: The response from the model, raw text and parsed components."""
        return self._call_api('evaluatePrompt', 'POST', query_params={}, body={'prompt': prompt, 'systemMessage': system_message, 'llmName': llm_name, 'maxTokens': max_tokens}, parse_type=LlmResponse, timeout=300)

    def render_feature_groups_for_llm(self, feature_group_ids: list, token_budget: int = None, include_definition: bool = True) -> LlmInput:
        """Encode feature groups as language model inputs.

        Args:
            feature_group_ids (list): List of feature groups to be encoded.
            token_budget (int): Enforce a given budget for each encoded feature group.
            include_definition (bool): Include the definition of the feature group in the encoding.

        Returns:
            LlmInput: LLM input object comprising of information about the feature groups with given IDs."""
        return self._call_api('renderFeatureGroupsForLLM', 'POST', query_params={}, body={'featureGroupIds': feature_group_ids, 'tokenBudget': token_budget, 'includeDefinition': include_definition}, parse_type=LlmInput)

    def get_llm_parameters(self, prompt: str, system_message: str = None, llm_name: str = None, max_tokens: int = None) -> LlmParameters:
        """Generate parameteres to the prompt using the given inputs

        Args:
            prompt (str): Prompt to use for generation.
            system_message (str): System message for models that support it.
            llm_name (str): Name of the underlying LLM to be used for generation. Should be one of 'gpt-4' or 'gpt-3.5-turbo'. Default is auto selection.
            max_tokens (int): Maximum number of tokens to generate. If set, the model will just stop generating after this token limit is reached.

        Returns:
            LlmParameters: The parameters for LLM using the given inputs."""
        return self._call_api('getLLMParameters', 'POST', query_params={}, body={'prompt': prompt, 'systemMessage': system_message, 'llmName': llm_name, 'maxTokens': max_tokens}, parse_type=LlmParameters, timeout=300)

    def create_document_retriever(self, project_id: str, name: str, feature_group_id: str, document_retriever_config: Union[dict, DocumentRetrieverConfig] = None) -> DocumentRetriever:
        """Returns a document retriever that stores embeddings for document chunks in a feature group.

        Document columns in the feature group are broken into chunks. For cases with multiple document columns, chunks from all columns are combined together to form a single chunk.


        Args:
            project_id (str): The ID of project that the vector store is created in.
            name (str): The name of the vector store.
            feature_group_id (str): The ID of the feature group that the vector store is associated with.
            document_retriever_config (DocumentRetrieverConfig): The configuration, including chunk_size and chunk_overlap_fraction, for document retrieval.

        Returns:
            DocumentRetriever: The newly created document retriever."""
        return self._call_api('createDocumentRetriever', 'POST', query_params={}, body={'projectId': project_id, 'name': name, 'featureGroupId': feature_group_id, 'documentRetrieverConfig': document_retriever_config}, parse_type=DocumentRetriever)

    def update_document_retriever(self, document_retriever_id: str, name: str = None, feature_group_id: str = None, document_retriever_config: Union[dict, DocumentRetrieverConfig] = None) -> DocumentRetriever:
        """Updates an existing document retriever.

        Args:
            document_retriever_id (str): The unique ID associated with the document retriever.
            name (str): The name group to update the document retriever with.
            feature_group_id (str): The ID of the feature group to update the document retriever with.
            document_retriever_config (DocumentRetrieverConfig): The configuration, including chunk_size and chunk_overlap_fraction, for document retrieval.

        Returns:
            DocumentRetriever: The updated document retriever."""
        return self._call_api('updateDocumentRetriever', 'POST', query_params={}, body={'documentRetrieverId': document_retriever_id, 'name': name, 'featureGroupId': feature_group_id, 'documentRetrieverConfig': document_retriever_config}, parse_type=DocumentRetriever)

    def create_document_retriever_version(self, document_retriever_id: str) -> DocumentRetrieverVersion:
        """Creates a document retriever version from the latest version of the feature group that the document retriever associated with.

        Args:
            document_retriever_id (str): The unique ID associated with the document retriever to create version with.

        Returns:
            DocumentRetrieverVersion: The newly created document retriever version."""
        return self._call_api('createDocumentRetrieverVersion', 'POST', query_params={}, body={'documentRetrieverId': document_retriever_id}, parse_type=DocumentRetrieverVersion)

    def delete_document_retriever(self, vector_store_id: str):
        """Delete a Document Retriever.

        Args:
            vector_store_id (str): A unique string identifier associated with the document retriever."""
        return self._call_api('deleteDocumentRetriever', 'DELETE', query_params={'vectorStoreId': vector_store_id})

    def lookup_document_retriever(self, document_retriever_id: str, query: str, deployment_token: str, filters: dict = None, limit: int = None, result_columns: list = None, max_words: int = None, num_retrieval_margin_words: int = None) -> List[DocumentRetrieverLookupResult]:
        """Lookup relevant documents from the document retriever deployed with given query.

        Args:
            document_retriever_id (str): A unique string identifier associated with the document retriever.
            query (str): The query to search for.
            deployment_token (str): A deployment token used to authenticate access to created vector store.
            filters (dict): A dictionary mapping column names to a list of values to restrict the retrieved search results.
            limit (int): If provided, will limit the number of results to the value specified.
            result_columns (list): If provided, will limit the column properties present in each result to those specified in this list.
            max_words (int): If provided, will limit the total number of words in the results to the value specified.
            num_retrieval_margin_words (int): If provided, will add this number of words from left and right of the returned chunks.

        Returns:
            list[DocumentRetrieverLookupResult]: The relevant documentation results found from the document retriever."""
        return self._call_api('lookupDocumentRetriever', 'POST', query_params={'deploymentToken': deployment_token}, body={'documentRetrieverId': document_retriever_id, 'query': query, 'filters': filters, 'limit': limit, 'resultColumns': result_columns, 'maxWords': max_words, 'numRetrievalMarginWords': num_retrieval_margin_words}, parse_type=DocumentRetrieverLookupResult)
