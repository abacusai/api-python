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
from copy import deepcopy
from functools import lru_cache
from typing import Any, Callable, Dict, List, Union
from uuid import uuid4

import pandas as pd
import requests
from packaging import version
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from .abacus_api import AbacusApi
from .agent import Agent
from .agent_conversation import AgentConversation
from .agent_data_execution_result import AgentDataExecutionResult
from .agent_version import AgentVersion
from .algorithm import Algorithm
from .annotation_config import AnnotationConfig
from .annotation_document import AnnotationDocument
from .annotation_entry import AnnotationEntry
from .annotations_status import AnnotationsStatus
from .api_class import (
    AgentClientType, AgentInterface, AlertActionConfig, AlertConditionConfig,
    ApiClass, ApiEnum, ApplicationConnectorDatasetConfig,
    ApplicationConnectorType, AttachmentParsingConfig, BatchPredictionArgs,
    Blob, BlobInput, CPUSize, DatasetDocumentProcessingConfig, DataType,
    DeploymentConversationType, DocumentProcessingConfig, EvalArtifactType,
    FeatureGroupExportConfig, ForecastingMonitorConfig,
    IncrementalDatabaseConnectorConfig, LLMName, MemorySize, MergeConfig,
    OperatorConfig, ParsingConfig, PredictionArguments, ProblemType,
    ProjectFeatureGroupConfig, PythonFunctionType, ResponseSection,
    SamplingConfig, Segment, StreamingConnectorDatasetConfig, TrainingConfig,
    VectorStoreConfig, WorkflowGraph, WorkflowGraphNode,
    WorkflowNodeTemplateConfig, get_clean_function_source_code_for_agent
)
from .api_class.abstract import get_clean_function_source_code, get_clean_function_source_code_for_agent, snake_case
from .api_class.ai_agents import WorkflowGraph, WorkflowNodeTemplateConfig
from .api_class.blob_input import Blob, BlobInput
from .api_class.enums import AgentInterface
from .api_class.segments import ResponseSection, Segment
from .api_client_utils import (
    INVALID_PANDAS_COLUMN_NAME_CHARACTERS, StreamingHandler, StreamType,
    _get_spark_incompatible_columns, clean_column_name,
    get_object_from_context, run
)
from .api_endpoint import ApiEndpoint
from .api_key import ApiKey
from .app_user_group import AppUserGroup
from .app_user_group_sign_in_token import AppUserGroupSignInToken
from .application_connector import ApplicationConnector
from .batch_prediction import BatchPrediction
from .batch_prediction_version import BatchPredictionVersion
from .batch_prediction_version_logs import BatchPredictionVersionLogs
from .chat_session import ChatSession
from .custom_loss_function import CustomLossFunction
from .custom_metric import CustomMetric
from .custom_metric_version import CustomMetricVersion
from .custom_train_function_info import CustomTrainFunctionInfo
from .data_metrics import DataMetrics
from .data_prep_logs import DataPrepLogs
from .database_connector import DatabaseConnector
from .database_connector_schema import DatabaseConnectorSchema
from .dataset import Dataset
from .dataset_column import DatasetColumn
from .dataset_version import DatasetVersion
from .dataset_version_logs import DatasetVersionLogs
from .deployment import Deployment
from .deployment_auth_token import DeploymentAuthToken
from .deployment_conversation import DeploymentConversation
from .deployment_conversation_export import DeploymentConversationExport
from .deployment_statistics import DeploymentStatistics
from .document_data import DocumentData
from .document_retriever import DocumentRetriever
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
from .external_application import ExternalApplication
from .external_invite import ExternalInvite
from .extracted_fields import ExtractedFields
from .feature import Feature
from .feature_distribution import FeatureDistribution
from .feature_group import FeatureGroup
from .feature_group_export import FeatureGroupExport
from .feature_group_export_config import FeatureGroupExportConfig
from .feature_group_export_download_url import FeatureGroupExportDownloadUrl
from .feature_group_row import FeatureGroupRow
from .feature_group_row_process import FeatureGroupRowProcess
from .feature_group_row_process_logs import FeatureGroupRowProcessLogs
from .feature_group_row_process_summary import FeatureGroupRowProcessSummary
from .feature_group_template import FeatureGroupTemplate
from .feature_group_version import FeatureGroupVersion
from .feature_group_version_logs import FeatureGroupVersionLogs
from .feature_importance import FeatureImportance
from .file_connector import FileConnector
from .file_connector_instructions import FileConnectorInstructions
from .file_connector_verification import FileConnectorVerification
from .function_logs import FunctionLogs
from .generated_pit_feature_config_option import GeneratedPitFeatureConfigOption
from .graph_dashboard import GraphDashboard
from .holdout_analysis import HoldoutAnalysis
from .holdout_analysis_version import HoldoutAnalysisVersion
from .inferred_feature_mappings import InferredFeatureMappings
from .llm_app import LlmApp
from .llm_execution_result import LlmExecutionResult
from .llm_generated_code import LlmGeneratedCode
from .llm_input import LlmInput
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
from .model_upload import ModelUpload
from .model_version import ModelVersion
from .model_version_feature_group_schema import ModelVersionFeatureGroupSchema
from .modification_lock_info import ModificationLockInfo
from .module import Module
from .monitor_alert import MonitorAlert
from .monitor_alert_version import MonitorAlertVersion
from .natural_language_explanation import NaturalLanguageExplanation
from .organization_group import OrganizationGroup
from .organization_search_result import OrganizationSearchResult
from .organization_secret import OrganizationSecret
from .page_data import PageData
from .pipeline import Pipeline
from .pipeline_step import PipelineStep
from .pipeline_step_version import PipelineStepVersion
from .pipeline_step_version_logs import PipelineStepVersionLogs
from .pipeline_version import PipelineVersion
from .pipeline_version_logs import PipelineVersionLogs
from .prediction_log_record import PredictionLogRecord
from .prediction_operator import PredictionOperator
from .prediction_operator_version import PredictionOperatorVersion
from .problem_type import ProblemType
from .project import Project
from .project_config import ProjectConfig
from .project_feature_group import ProjectFeatureGroup
from .project_validation import ProjectValidation
from .python_function import PythonFunction
from .python_plot_function import PythonPlotFunction
from .realtime_monitor import RealtimeMonitor
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
from .web_page_response import WebPageResponse
from .web_search_response import WebSearchResponse
from .webhook import Webhook
from .workflow_node_template import WorkflowNodeTemplate


DEFAULT_SERVER = 'https://api.abacus.ai'


_request_context = threading.local()


def _is_json_serializable(object: Any):
    """
    Tests if an object is JSON serializable.

    Args:
        object (any): The object to test.

    Returns:
        bool: True if the object is JSON serializable, False otherwise.
    """
    try:
        json.dumps(object)
        return True
    except TypeError:
        return False


async def sse_asynchronous_generator(endpoint: str, headers: dict, body: dict):
    try:
        import aiohttp
    except Exception:
        raise Exception('Please install aiohttp to use this functionality')

    async with aiohttp.request('POST', endpoint, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=0)) as response:
        async for line in response.content:
            if line:
                streamed_responses = line.decode('utf-8').split('\n\n')
                for resp in streamed_responses:
                    if resp:
                        resp = resp.strip()
                        if resp:
                            resp = json.loads(resp)
                            resp = {snake_case(
                                key): value for key, value in resp.items()}
                            if 'ping' in resp:
                                continue
                            yield resp


def _requests_retry_session(retries=5, backoff_factor=0.1, status_forcelist=(502, 503, 504), session=None, retry_500: bool = False):
    session = session or requests.Session()
    if retry_500:
        status_forcelist = (500, *status_forcelist)
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


class AgentResponse:
    """
    Response object for agent to support attachments, section data and normal data
    """

    def __init__(self, *args, **kwargs):
        self.data_list = []
        self.section_data_list = []
        for arg in args:
            if isinstance(arg, Blob) or _is_json_serializable(arg) or isinstance(arg, (ResponseSection, Segment)):
                self.data_list.append(arg)
            else:
                raise Exception(
                    'AgentResponse can only contain Blob, ResponseSection or json serializable objects if key is not provided')

        for key, value in kwargs.items():
            self.section_data_list.append({key: value})

    def __getstate__(self):
        """Return state values to be pickled."""
        return {'data_list': self.data_list, 'section_data_list': self.section_data_list}

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.data_list = state['data_list']
        self.section_data_list = state['section_data_list']

    def to_dict(self):
        """
        Get a dict representation of the response object
        """
        result = {}
        if self.data_list:
            result['data_list'] = self.data_list
        for section_data in self.section_data_list:
            for k, v in section_data.items():
                if isinstance(v, ResponseSection):
                    result[k] = v.to_dict()
                else:
                    result[k] = v
        return result

    def __getattr__(self, item):
        for section_data in self.section_data_list:
            for k, v in section_data.items():
                if k == item:
                    return v
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{item}'")


class ToolResponse(AgentResponse):
    """
    Response object for tool to support non-text response sections
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
        request_id (str): The request id
    """

    def __init__(self, message: str, http_status: int, exception: str = None, request_id: str = None):
        self.message = message
        self.http_status = http_status
        self.exception = exception or 'ApiException'
        self.request_id = request_id

    def __str__(self):
        return f'{self.exception}({self.http_status}): {self.message}'


class MissingParameterError(ApiException):
    """
    Missing parameter error raised by APIs. This is usually raised when a required parameter is missing in the request.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class InvalidParameterError(ApiException):
    """
    Raised when an invalid parameter is provided.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class InvalidEnumParameterError(ApiException):
    """
    Raised when an invalid enumeration parameter is provided.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class PaymentMethodRequired(ApiException):
    """
    Raised when a payment method is required.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class InvalidRequest(ApiException):
    """
    Invalid request error raised by APIs. This is usually raised when the request is invalid or malformed.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class GenericPermissionDeniedError(ApiException):
    """
    Generic permission denied error raised by APIs. This is usually raised when permission is denied.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class PermissionDeniedError(ApiException):
    """
    Permission denied error raised by APIs. This is usually raised when a specific operation is not permitted.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class PaidPlanRequired(ApiException):
    """
    Paid plan required error raised by APIs. This is usually raised when a paid plan is required for an operation.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class Generic404Error(ApiException):
    """
    Generic 404 error raised by APIs. This is usually raised when a resource is not found.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class DataNotFoundError(ApiException):
    """
    Data not found error raised by APIs. This is usually raised when specific data is not found.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class MethodNotAllowed(ApiException):
    """
    Method not allowed error raised by APIs. This is usually raised when a specific http method is not allowed for a resource.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class RequestTimeoutError(ApiException):
    """
    Exception raised when a timeout occurs during API request.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class ConflictError(ApiException):
    """
    Exception raised when a conflict occurs during API request.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class AlreadyExistsError(ApiException):
    """
    Exception raised when the entity being created already exists.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class NotReadyError(ApiException):
    """
    Not ready exception raised by APIs. This is usually raised when the operation requested is not ready.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class FailedDependencyError(ApiException):
    """
    Failed dependency exception raised by APIs. This is usually raised when the operation failed due to a dependency error.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class TooManyRequestsError(ApiException):
    """
    Too many requests exception raised by APIs. This is usually raised when the rate limit for requests has been exceeded.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class InstanceNotModifiedError(ApiException):
    """
    InstanceNotModifiedError exception raised by APIs. This is usually raised when an instance is not modified.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class GatewayTimeoutError(ApiException):
    """
    Gateway timeout error raised by APIs. This is usually raised when a request
    times out waiting for a response from the gateway.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class InternalServerError(ApiException):
    """
    Internal server error raised by APIs. This is usually raised when the server
    encounters an unexpected error while processing the request.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """


class UnsuccessfulTrigger(ApiException):
    """
    Error class to indicate that the trigger is unsuccessful for autonomous agents
    and avoids execution of the further workflow.

    Args:
        message (str): The error message
        http_status (int): The https status code raised by the server
        exception (str): The exception class raised by the server
        request_id (str): The request id
    """

    def __init__(self, message: str):
        super().__init__(message, 430)


class _ApiExceptionFactory:
    """
    Factory class to build exceptions raised by APIs

    """
    @classmethod
    def from_response(cls, message: str, http_status: int, exception: str = None, request_id: str = None):
        """
        Creates an appropriate exception instance based on HTTP response data.
        """
        if http_status == 504:
            message = 'Gateway timeout, please try again later'
            return GatewayTimeoutError(message, http_status, exception, request_id)
        elif http_status == 429:
            message = 'Too many requests. Please slow down your API requests'
            return TooManyRequestsError(message, http_status, exception, request_id)
        elif http_status > 502 and http_status not in (501, 503):
            message = 'Internal Server Error, please contact dev@abacus.ai for support'
            return InternalServerError(message, http_status, exception, request_id)

        if exception == 'TimeoutError':
            exception = 'RequestTimeoutError'
        try:
            class_type = globals()[exception]
        except KeyError:
            class_type = ApiException
        return class_type(message, http_status, exception, request_id)


class BaseApiClient:
    """
    Abstract Base API Client

    Args:
        api_key (str): The api key to use as authentication to the server
        server (str): The base server url to use to send API requets to
        client_options (ClientOptions): Optional API client configurations
        skip_version_check (bool): If true, will skip checking the server's current API version on initializing the client
    """
    client_version = '1.4.44'

    def __init__(self, api_key: str = None, server: str = None, client_options: ClientOptions = None, skip_version_check: bool = False, include_tb: bool = False):
        self.api_key = api_key
        if self.api_key is None:
            self.api_key = get_object_from_context(
                self, _request_context, 'ABACUS_API_KEY', str) or os.getenv('ABACUS_API_KEY')
        self.notebook_id = os.getenv('ABACUS_NOTEBOOK_ID')
        self.pipeline_id = os.getenv('ABACUS_PIPELINE_ID')
        self.web_version = None
        self.api_endpoint = None
        self.prediction_endpoint = None
        self.proxy_endpoint = None
        if not client_options:
            client_options = ClientOptions(server=os.getenv(
                'ABACUS_API_SERVER')) if os.getenv('ABACUS_API_SERVER') else ClientOptions()
        self.client_options = client_options
        self.server = server or self.client_options.server
        self.cache_scope = None
        self._cache = {}
        # Deprecated
        self.service_discovery_url = None
        # Connection and version check
        if self.api_key is not None:
            try:
                endpoint_info = self._call_api('getApiEndpoint', 'GET')
                self.prediction_endpoint = endpoint_info['predictEndpoint']
                self.proxy_endpoint = endpoint_info.get('proxyEndpoint')
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
        # Modify traceback
        if not include_tb:
            excepthook = sys.excepthook

            def modify_tb_excepthook(type, value, tb):
                import abacusai
                pylibdir = os.path.dirname(abacusai.__file__)
                tb_itr = tb
                while not (tb_itr.tb_next is None or tb_itr.tb_next.tb_frame.f_code.co_filename.startswith(pylibdir)):
                    tb_itr = tb_itr.tb_next
                tb_itr.tb_next = None
                excepthook(type, value, tb)
            sys.excepthook = modify_tb_excepthook

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

    def _get_proxy_endpoint(self, deployment_id: str, deployment_token: str):
        if self.proxy_endpoint:
            return self.proxy_endpoint
        global _cached_endpoints
        cache_key = deployment_id + '|' + deployment_token
        endpoint_info = _cached_endpoints.get(cache_key)
        if not endpoint_info:
            endpoint_info = self._call_api('getApiEndpoint', 'GET', query_params={
                                           'deploymentId': deployment_id, 'deploymentToken': deployment_token})
            _cached_endpoints[cache_key] = endpoint_info
        return endpoint_info.get('proxyEndpoint')

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
                class_obj_dict = val.to_dict()
                self._clean_api_objects(class_obj_dict)
                obj[key] = class_obj_dict
            elif isinstance(val, list):
                for index, list_val in enumerate(val):
                    if isinstance(list_val, dict):
                        self._clean_api_objects(list_val)
                    elif isinstance(list_val, ApiClass):
                        obj[key][index] = list_val.to_dict()
                        self._clean_api_objects(obj[key][index])
                    elif isinstance(list_val, ApiEnum):
                        obj[key][index] = list_val.value
                    else:
                        obj[key][index] = list_val
            elif isinstance(val, dict):
                self._clean_api_objects(val)
            elif isinstance(val, ApiEnum):
                obj[key] = val.value
            elif callable(val):
                try:
                    obj[key] = get_clean_function_source_code(val)
                except OSError:
                    raise OSError(
                        f'Could not get source for function {key}. Please pass a stringified version of this function when the function is defined in a shell environment.')

    def _call_api(
            self, action, method, query_params=None,
            body=None, files=None, parse_type=None, streamable_response=False, server_override=None, timeout=None, retry_500: bool = False, data=None):
        user_agent = f'python-abacusai-{self.client_version}'
        if self.notebook_id:
            user_agent = user_agent + f' | notebookId: {self.notebook_id}'
        if os.getenv('ABACUS_ARTIFACT_ID'):
            user_agent = user_agent + \
                f' | artifactId: {os.getenv("ABACUS_ARTIFACT_ID")}'
        headers = {'apiKey': self.api_key, 'clientVersion': self.client_version,
                   'User-Agent': user_agent, 'notebookId': self.notebook_id, 'pipelineId': self.pipeline_id}
        url = (server_override or self.server) + '/api/v0/' + action
        copied_query_params = deepcopy(query_params)
        copied_body = deepcopy(body)
        self._clean_api_objects(copied_query_params)
        self._clean_api_objects(copied_body)
        if self.service_discovery_url:
            discovered_url = _discover_service_url(self.service_discovery_url, self.client_version, copied_query_params.get(
                'deploymentId') if copied_query_params else None, copied_query_params.get('deploymentToken') if copied_query_params else None)
            if discovered_url:
                url = discovered_url + '/api/' + action
        response = self._request(url, method, query_params=copied_query_params, headers=headers, body=copied_body,
                                 files=files, stream=streamable_response, timeout=timeout, retry_500=retry_500, data=data)

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
                f"_call_api caught an exception {e} in processing json_data {json_data}. API call url method body: {url} {method} '{json.dumps(copied_body)}'")
            error_message = response.text
        if not success:
            if response.status_code == 404 and not self.client_options.exception_on_404:
                return None
            if request_id is None and response.headers:
                request_id = response.headers.get('x-request-id')
            if request_id:
                error_message += f". Request ID: {request_id}"
            raise _ApiExceptionFactory.from_response(
                error_message, response.status_code, error_type, request_id)
        return result

    def _proxy_request(self, name: str, method: str = 'POST', query_params: dict = None, body: dict = None, data: dict = None, files=None, parse_type=None, is_sync: bool = False, streamable_response: bool = False):
        headers = {'APIKEY': self.api_key}
        deployment_id = os.getenv('ABACUS_EXEC_SERVICE_DEPLOYMENT_ID')
        if deployment_id:
            query_params = {**(query_params or {}),
                            'environmentDeploymentId': deployment_id}
        caller = self._get_agent_caller()
        request_id = self._get_agent_app_request_id()
        if caller and request_id:
            if get_object_from_context(self, _request_context, 'is_agent', bool):
                query_params = {**(query_params or {}), 'isAgent': True}
            if get_object_from_context(self, _request_context, 'is_agent_api', bool):
                query_params = {**(query_params or {}), 'isAgentApi': True}
        endpoint = self.proxy_endpoint
        if endpoint is None:
            raise Exception(
                'API not supported, Please contact Abacus.ai support')
        result = None
        error_json = {}
        status_code = 200
        response = None
        request_id = None

        query_params = deepcopy(query_params)
        body = deepcopy(body)
        self._clean_api_objects(query_params)
        self._clean_api_objects(body)
        if data:
            data = deepcopy(data)
            self._clean_api_objects(data)

        try:
            if is_sync:
                sync_api_endpoint = f'{endpoint}/api/{name}'
                response = self._request(url=sync_api_endpoint, method=method, query_params=query_params,
                                         headers=headers, body=body, data=data, files=files, stream=streamable_response)
                status_code = response.status_code
                if streamable_response and status_code == 200:
                    return response.raw
                response = response.json()
                if response.get('success'):
                    result = response.get('result')
                    return self._build_class(parse_type, result) if parse_type else result
                error_json = response
            else:
                create_request_endpoint = f'{endpoint}/api/create{name}Request'
                status_request_endpoint = f'{endpoint}/api/get{name}Status'
                create_request = self._request(url=create_request_endpoint, method='PUT' if files else 'POST',
                                               query_params=query_params, headers=headers, body=body or data, files=files)
                status_code = create_request.status_code
                if status_code == 200:
                    request_id = create_request.json()['request_id']
                    response, status_code = self._status_poll(url=status_request_endpoint, wait_states={
                                                              'PENDING'}, method='POST', body={'request_id': request_id}, headers=headers)
                    if response['status'] == 'FAILED':
                        error_json = response.get('result')
                    else:
                        result = response['result']
                        if result.get('success'):
                            return self._build_class(parse_type, result.get('result'))
                else:
                    error_json = create_request.json()
        except Exception:
            pass

        error_message = error_json.get('error') or 'Unknown error'
        error_type = error_json.get('errorType')
        raise _ApiExceptionFactory.from_response(
            error_message, status_code, error_type, request_id)

    def _build_class(self, return_class, values):
        if values is None or values == {}:
            return None
        if isinstance(values, list):
            return [self._build_class(return_class, val) for val in values if val is not None]
        if issubclass(return_class, ApiClass):
            return return_class.from_dict(values)
        type_inputs = inspect.signature(return_class.__init__).parameters
        return return_class(self, **{key: val for key, val in values.items() if key in type_inputs})

    def _request(self, url, method, query_params=None, headers=None,
                 body=None, files=None, stream=False, timeout=None, retry_500: bool = False, data=None):
        _session = _requests_retry_session(retry_500=retry_500)
        if method == 'GET':
            cleaned_params = {key: json.dumps(val) if (isinstance(val, list) or isinstance(
                val, dict)) else val for key, val in query_params.items()} if query_params else query_params
            return _session.get(url, params=cleaned_params, headers=headers, stream=stream)
        elif method == 'POST':
            return _session.post(url, params=query_params, json=body, headers=headers, files=files, timeout=timeout or 600, data=data)
        elif method == 'PUT':
            return _session.put(url, params=query_params, data=body, headers=headers, files=files, timeout=timeout or 600)
        elif method == 'PATCH':
            return _session.patch(url, params=query_params, json=body, headers=headers, files=files, timeout=timeout or 600)
        elif method == 'DELETE':
            return _session.delete(url, params=query_params, data=body, headers=headers)
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

        incompatible_columns, compatible_pd_dtypes = _get_spark_incompatible_columns(
            df)
        if incompatible_columns:
            error_message = "The following columns have incompatible data types:\n"
            for col_name, col_dtype in incompatible_columns:
                error_message += f" - '{col_name}' (type: {col_dtype})\n"
            error_message += f"Supported data types are: {', '.join(sorted(compatible_pd_dtypes))}\n"
            error_message += "Please cast these columns to a supported data type and try again.\n"
            raise ValueError(error_message)

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
        """Retrieves a list of all platform users in the organization, including pending users who have been invited.

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

    def validate_project(self, project_id: str, feature_group_ids: List = None) -> ProjectValidation:
        """Validates that the specified project has all required feature group types for its use case and that all required feature columns are set.

        Args:
            project_id (str): The unique ID associated with the project.
            feature_group_ids (List): The list of feature group IDs to validate.

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

    def describe_project_feature_group(self, project_id: str, feature_group_id: str) -> ProjectFeatureGroup:
        """Describe a feature group associated with a project

        Args:
            project_id (str): The unique ID associated with the project.
            feature_group_id (str): The unique ID associated with the feature group.

        Returns:
            ProjectFeatureGroup: The project feature group object."""
        return self._call_api('describeProjectFeatureGroup', 'GET', query_params={'projectId': project_id, 'featureGroupId': feature_group_id}, parse_type=ProjectFeatureGroup)

    def list_project_feature_groups(self, project_id: str, filter_feature_group_use: str = None, limit: int = 100, start_after_id: str = None) -> List[ProjectFeatureGroup]:
        """List all the feature groups associated with a project

        Args:
            project_id (str): The unique ID associated with the project.
            filter_feature_group_use (str): The feature group use filter, when given as an argument only allows feature groups present in this project to be returned if they are of the given use. Possible values are: 'USER_CREATED', 'BATCH_PREDICTION_OUTPUT'.
            limit (int): The maximum number of feature groups to be retrieved.
            start_after_id (str): An offset parameter to exclude all feature groups up to a specified ID.

        Returns:
            list[ProjectFeatureGroup]: All the Feature Groups in a project."""
        return self._call_api('listProjectFeatureGroups', 'GET', query_params={'projectId': project_id, 'filterFeatureGroupUse': filter_feature_group_use, 'limit': limit, 'startAfterId': start_after_id}, parse_type=ProjectFeatureGroup)

    def list_python_function_feature_groups(self, name: str, limit: int = 100) -> List[FeatureGroup]:
        """List all the feature groups associated with a python function.

        Args:
            name (str): The name used to identify the Python function.
            limit (int): The maximum number of feature groups to be retrieved.

        Returns:
            list[FeatureGroup]: All the feature groups associated with the specified Python function ID."""
        return self._call_api('listPythonFunctionFeatureGroups', 'GET', query_params={'name': name, 'limit': limit}, parse_type=FeatureGroup)

    def get_execute_feature_group_operation_result_part_count(self, feature_group_operation_run_id: str) -> int:
        """Gets the number of parts in the result of the execution of fg operation

        Args:
            feature_group_operation_run_id (str): The unique ID associated with the execution."""
        return self._call_api('getExecuteFeatureGroupOperationResultPartCount', 'GET', query_params={'featureGroupOperationRunId': feature_group_operation_run_id})

    def download_execute_feature_group_operation_result_part_chunk(self, feature_group_operation_run_id: str, part: int, offset: int = 0, chunk_size: int = 10485760) -> io.BytesIO:
        """Downloads a chunk of the result of the execution of feature group operation

        Args:
            feature_group_operation_run_id (str): The unique ID associated with the execution.
            part (int): The part number of the result
            offset (int): The offset in the part
            chunk_size (int): The size of the chunk"""
        return self._call_api('downloadExecuteFeatureGroupOperationResultPartChunk', 'GET', query_params={'featureGroupOperationRunId': feature_group_operation_run_id, 'part': part, 'offset': offset, 'chunkSize': chunk_size}, streamable_response=True)

    def update_feature_group_version_limit(self, feature_group_id: str, version_limit: int) -> FeatureGroup:
        """Updates the version limit for the feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            version_limit (int): The maximum number of versions permitted for the feature group. Once this limit is exceeded, the oldest versions will be purged in a First-In-First-Out (FIFO) order.

        Returns:
            FeatureGroup: The updated feature group."""
        return self._call_api('updateFeatureGroupVersionLimit', 'GET', query_params={'featureGroupId': feature_group_id, 'versionLimit': version_limit}, parse_type=FeatureGroup)

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

    def get_feature_group_version_metrics(self, feature_group_version: str, selected_columns: List = None, include_charts: bool = False, include_statistics: bool = True) -> DataMetrics:
        """Get metrics for a specific feature group version.

        Args:
            feature_group_version (str): A unique string identifier associated with the feature group version.
            selected_columns (List): A list of columns to order first.
            include_charts (bool): A flag indicating whether charts should be included in the response. Default is false.
            include_statistics (bool): A flag indicating whether statistics should be included in the response. Default is true.

        Returns:
            DataMetrics: The metrics for the specified feature group version."""
        return self._call_api('getFeatureGroupVersionMetrics', 'GET', query_params={'featureGroupVersion': feature_group_version, 'selectedColumns': selected_columns, 'includeCharts': include_charts, 'includeStatistics': include_statistics}, parse_type=DataMetrics)

    def get_feature_group_version_logs(self, feature_group_version: str) -> FeatureGroupVersionLogs:
        """Retrieves the feature group materialization logs.

        Args:
            feature_group_version (str): The unique version ID of the feature group version.

        Returns:
            FeatureGroupVersionLogs: The logs for the specified feature group version."""
        return self._call_api('getFeatureGroupVersionLogs', 'GET', query_params={'featureGroupVersion': feature_group_version}, parse_type=FeatureGroupVersionLogs)

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

    def get_dataset_version_metrics(self, dataset_version: str, selected_columns: List = None, include_charts: bool = False, include_statistics: bool = True) -> DataMetrics:
        """Get metrics for a specific dataset version.

        Args:
            dataset_version (str): A unique string identifier associated with the dataset version.
            selected_columns (List): A list of columns to order first.
            include_charts (bool): A flag indicating whether charts should be included in the response. Default is false.
            include_statistics (bool): A flag indicating whether statistics should be included in the response. Default is true.

        Returns:
            DataMetrics: The metrics for the specified Dataset version."""
        return self._call_api('getDatasetVersionMetrics', 'GET', query_params={'datasetVersion': dataset_version, 'selectedColumns': selected_columns, 'includeCharts': include_charts, 'includeStatistics': include_statistics}, parse_type=DataMetrics)

    def update_dataset_version_limit(self, dataset_id: str, version_limit: int) -> Dataset:
        """Updates the version limit for the specified dataset.

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            version_limit (int): The maximum number of versions permitted for the feature group. Once this limit is exceeded, the oldest versions will be purged in a First-In-First-Out (FIFO) order.

        Returns:
            Dataset: The updated dataset."""
        return self._call_api('updateDatasetVersionLimit', 'GET', query_params={'datasetId': dataset_id, 'versionLimit': version_limit}, parse_type=Dataset)

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

    def list_database_connector_objects(self, database_connector_id: str, fetch_raw_data: bool = False) -> list:
        """Lists querable objects in the database connector.

        Args:
            database_connector_id (str): Unique string identifier for the database connector.
            fetch_raw_data (bool): If true, return unfiltered objects."""
        return self._call_api('listDatabaseConnectorObjects', 'GET', query_params={'databaseConnectorId': database_connector_id, 'fetchRawData': fetch_raw_data})

    def get_database_connector_object_schema(self, database_connector_id: str, object_name: str = None, fetch_raw_data: bool = False) -> DatabaseConnectorSchema:
        """Get the schema of an object in an database connector.

        Args:
            database_connector_id (str): Unique string identifier for the database connector.
            object_name (str): Unique identifier for the object in the external system.
            fetch_raw_data (bool): If true, return unfiltered list of columns.

        Returns:
            DatabaseConnectorSchema: The schema of the object."""
        return self._call_api('getDatabaseConnectorObjectSchema', 'GET', query_params={'databaseConnectorId': database_connector_id, 'objectName': object_name, 'fetchRawData': fetch_raw_data}, parse_type=DatabaseConnectorSchema)

    def query_database_connector(self, database_connector_id: str, query: str) -> list:
        """Runs a query in the specified database connector.

        Args:
            database_connector_id (str): A unique string identifier for the database connector.
            query (str): The query to be run in the database connector."""
        return self._call_api('queryDatabaseConnector', 'GET', query_params={'databaseConnectorId': database_connector_id, 'query': query})

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

    def get_connector_auth(self, service: Union[ApplicationConnectorType, str] = None, application_connector_id: str = None, scopes: List = None, is_database_connector: bool = None) -> ApplicationConnector:
        """Get the authentication details for a given connector. For user level connectors, the service is required. For org level connectors, the application_connector_id is required.

        Args:
            service (ApplicationConnectorType): The service name.
            application_connector_id (str): The unique ID associated with the connector.
            scopes (List): The scopes to request for the connector.
            is_database_connector (bool): Whether the connector is a database connector.

        Returns:
            ApplicationConnector: The application connector with the authentication details."""
        return self._call_api('getConnectorAuth', 'GET', query_params={'service': service, 'applicationConnectorId': application_connector_id, 'scopes': scopes, 'isDatabaseConnector': is_database_connector}, parse_type=ApplicationConnector)

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

    def get_dataset_version_logs(self, dataset_version: str) -> DatasetVersionLogs:
        """Retrieves the dataset import logs.

        Args:
            dataset_version (str): The unique version ID of the dataset version.

        Returns:
            DatasetVersionLogs: The logs for the specified dataset version."""
        return self._call_api('getDatasetVersionLogs', 'GET', query_params={'datasetVersion': dataset_version}, parse_type=DatasetVersionLogs)

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
        return self._proxy_request('getDocstoreImage', 'GET', query_params={'docId': doc_id, 'maxWidth': max_width, 'maxHeight': max_height}, is_sync=True, streamable_response=True)

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

    def get_model_metrics(self, model_id: str, model_version: str = None, return_graphs: bool = False, validation: bool = False) -> ModelMetrics:
        """Retrieves metrics for all the algorithms trained in this model version.

        If only the model's unique identifier (model_id) is specified, the latest trained version of the model (model_version) is used.


        Args:
            model_id (str): Unique string identifier for the model.
            model_version (str): Version of the model.
            return_graphs (bool): If true, will return the information used for the graphs on the model metrics page such as PR Curve per label.
            validation (bool): If true, will return the validation metrics instead of the test metrics.

        Returns:
            ModelMetrics: An object containing the model metrics and explanations for what each metric means."""
        return self._call_api('getModelMetrics', 'GET', query_params={'modelId': model_id, 'modelVersion': model_version, 'returnGraphs': return_graphs, 'validation': validation}, parse_type=ModelMetrics)

    def get_feature_group_schemas_for_model_version(self, model_version: str) -> List[ModelVersionFeatureGroupSchema]:
        """Gets the schema (including feature mappings) for all feature groups used in the model version.

        Args:
            model_version (str): Unique string identifier for the version of the model.

        Returns:
            list[ModelVersionFeatureGroupSchema]: List of schema for all feature groups used in the model version."""
        return self._call_api('getFeatureGroupSchemasForModelVersion', 'GET', query_params={'modelVersion': model_version}, parse_type=ModelVersionFeatureGroupSchema)

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

    def get_training_logs(self, model_version: str, stdout: bool = False, stderr: bool = False) -> FunctionLogs:
        """Returns training logs for the model.

        Args:
            model_version (str): The unique version ID of the model version.
            stdout (bool): Set True to get info logs.
            stderr (bool): Set True to get error logs.

        Returns:
            FunctionLogs: A function logs object."""
        return self._call_api('getTrainingLogs', 'GET', query_params={'modelVersion': model_version, 'stdout': stdout, 'stderr': stderr}, parse_type=FunctionLogs)

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

    def list_model_monitors(self, project_id: str, limit: int = None) -> List[ModelMonitor]:
        """Retrieves the list of model monitors in the specified project.

        Args:
            project_id (str): Unique string identifier associated with the project.
            limit (int): Maximum number of model monitors to return. We'll have internal limit if not set.

        Returns:
            list[ModelMonitor]: A list of model monitors."""
        return self._call_api('listModelMonitors', 'GET', query_params={'projectId': project_id, 'limit': limit}, parse_type=ModelMonitor)

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

    def list_holdout_analysis(self, project_id: str, model_id: str = None) -> List[HoldoutAnalysis]:
        """List holdout analyses for a project. Optionally, filter by model.

        Args:
            project_id (str): ID of the project to list holdout analyses for
            model_id (str): (optional) ID of the model to filter by

        Returns:
            list[HoldoutAnalysis]: The holdout analyses"""
        return self._call_api('listHoldoutAnalysis', 'GET', query_params={'projectId': project_id, 'modelId': model_id}, parse_type=HoldoutAnalysis)

    def describe_holdout_analysis(self, holdout_analysis_id: str) -> HoldoutAnalysis:
        """Get a holdout analysis.

        Args:
            holdout_analysis_id (str): ID of the holdout analysis to get

        Returns:
            HoldoutAnalysis: The holdout analysis"""
        return self._call_api('describeHoldoutAnalysis', 'GET', query_params={'holdoutAnalysisId': holdout_analysis_id}, parse_type=HoldoutAnalysis)

    def list_holdout_analysis_versions(self, holdout_analysis_id: str) -> List[HoldoutAnalysisVersion]:
        """List holdout analysis versions for a holdout analysis.

        Args:
            holdout_analysis_id (str): ID of the holdout analysis to list holdout analysis versions for

        Returns:
            list[HoldoutAnalysisVersion]: The holdout analysis versions"""
        return self._call_api('listHoldoutAnalysisVersions', 'GET', query_params={'holdoutAnalysisId': holdout_analysis_id}, parse_type=HoldoutAnalysisVersion)

    def describe_holdout_analysis_version(self, holdout_analysis_version: str, get_metrics: bool = False) -> HoldoutAnalysisVersion:
        """Get a holdout analysis version.

        Args:
            holdout_analysis_version (str): ID of the holdout analysis version to get
            get_metrics (bool): (optional) Whether to get the metrics for the holdout analysis version

        Returns:
            HoldoutAnalysisVersion: The holdout analysis version"""
        return self._call_api('describeHoldoutAnalysisVersion', 'GET', query_params={'holdoutAnalysisVersion': holdout_analysis_version, 'getMetrics': get_metrics}, parse_type=HoldoutAnalysisVersion)

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

    def list_monitor_alerts_for_monitor(self, model_monitor_id: str = None, realtime_monitor_id: str = None) -> List[MonitorAlert]:
        """Retrieves the list of monitor alerts for a specified monitor. One of the model_monitor_id or realtime_monitor_id is required but not both.

        Args:
            model_monitor_id (str): The unique ID associated with the model monitor.
            realtime_monitor_id (str): The unique ID associated with the real-time monitor.

        Returns:
            list[MonitorAlert]: A list of monitor alerts."""
        return self._call_api('listMonitorAlertsForMonitor', 'GET', query_params={'modelMonitorId': model_monitor_id, 'realtimeMonitorId': realtime_monitor_id}, parse_type=MonitorAlert)

    def list_monitor_alert_versions_for_monitor_version(self, model_monitor_version: str) -> List[MonitorAlertVersion]:
        """Retrieves the list of monitor alert versions for a specified monitor instance.

        Args:
            model_monitor_version (str): The unique ID associated with the model monitor.

        Returns:
            list[MonitorAlertVersion]: A list of monitor alert versions."""
        return self._call_api('listMonitorAlertVersionsForMonitorVersion', 'GET', query_params={'modelMonitorVersion': model_monitor_version}, parse_type=MonitorAlertVersion)

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

    def describe_prediction_operator(self, prediction_operator_id: str) -> PredictionOperator:
        """Describe an existing prediction operator.

        Args:
            prediction_operator_id (str): The unique ID of the prediction operator.

        Returns:
            PredictionOperator: The requested prediction operator object."""
        return self._call_api('describePredictionOperator', 'GET', query_params={'predictionOperatorId': prediction_operator_id}, parse_type=PredictionOperator)

    def list_prediction_operators(self, project_id: str) -> List[PredictionOperator]:
        """List all the prediction operators inside a project.

        Args:
            project_id (str): The unique ID of the project.

        Returns:
            list[PredictionOperator]: A list of prediction operator objects."""
        return self._call_api('listPredictionOperators', 'GET', query_params={'projectId': project_id}, parse_type=PredictionOperator)

    def list_prediction_operator_versions(self, prediction_operator_id: str) -> List[PredictionOperatorVersion]:
        """List all the prediction operator versions for a prediction operator.

        Args:
            prediction_operator_id (str): The unique ID of the prediction operator.

        Returns:
            list[PredictionOperatorVersion]: A list of prediction operator version objects."""
        return self._call_api('listPredictionOperatorVersions', 'GET', query_params={'predictionOperatorId': prediction_operator_id}, parse_type=PredictionOperatorVersion)

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

    def get_prediction_logs_records(self, deployment_id: str, limit: int = 10, last_log_request_id: str = '', last_log_timestamp: int = None) -> List[PredictionLogRecord]:
        """Retrieves the prediction request IDs for the most recent predictions made to the deployment.

        Args:
            deployment_id (str): The unique identifier of a deployment created under the project.
            limit (int): The number of prediction log entries to retrieve up to the specified limit.
            last_log_request_id (str): The request ID of the last log entry to retrieve.
            last_log_timestamp (int): A Unix timestamp in milliseconds specifying the timestamp for the last log entry.

        Returns:
            list[PredictionLogRecord]: A list of prediction log records."""
        return self._call_api('getPredictionLogsRecords', 'GET', query_params={'deploymentId': deployment_id, 'limit': limit, 'lastLogRequestId': last_log_request_id, 'lastLogTimestamp': last_log_timestamp}, parse_type=PredictionLogRecord)

    def list_deployment_alerts(self, deployment_id: str) -> List[MonitorAlert]:
        """List the monitor alerts associated with the deployment id.

        Args:
            deployment_id (str): Unique string identifier for the deployment.

        Returns:
            list[MonitorAlert]: An array of deployment alerts."""
        return self._call_api('listDeploymentAlerts', 'GET', query_params={'deploymentId': deployment_id}, parse_type=MonitorAlert)

    def list_realtime_monitors(self, project_id: str) -> List[RealtimeMonitor]:
        """List the real-time monitors associated with the deployment id.

        Args:
            project_id (str): Unique string identifier for the deployment.

        Returns:
            list[RealtimeMonitor]: An array of real-time monitors."""
        return self._call_api('listRealtimeMonitors', 'GET', query_params={'projectId': project_id}, parse_type=RealtimeMonitor)

    def describe_realtime_monitor(self, realtime_monitor_id: str) -> RealtimeMonitor:
        """Get the real-time monitor associated with the real-time monitor id.

        Args:
            realtime_monitor_id (str): Unique string identifier for the real-time monitor.

        Returns:
            RealtimeMonitor: Object describing the real-time monitor."""
        return self._call_api('describeRealtimeMonitor', 'GET', query_params={'realtimeMonitorId': realtime_monitor_id}, parse_type=RealtimeMonitor)

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

    def list_refresh_policies(self, project_id: str = None, dataset_ids: List = [], feature_group_id: str = None, model_ids: List = [], deployment_ids: List = [], batch_prediction_ids: List = [], model_monitor_ids: List = [], notebook_ids: List = []) -> List[RefreshPolicy]:
        """List the refresh policies for the organization. If no filters are specified, all refresh policies are returned.

        Args:
            project_id (str): Project ID for which we wish to see the refresh policies attached.
            dataset_ids (List): Comma-separated list of Dataset IDs.
            feature_group_id (str): Feature Group ID for which we wish to see the refresh policies attached.
            model_ids (List): Comma-separated list of Model IDs.
            deployment_ids (List): Comma-separated list of Deployment IDs.
            batch_prediction_ids (List): Comma-separated list of Batch Prediction IDs.
            model_monitor_ids (List): Comma-separated list of Model Monitor IDs.
            notebook_ids (List): Comma-separated list of Notebook IDs.

        Returns:
            list[RefreshPolicy]: List of all refresh policies in the organization."""
        return self._call_api('listRefreshPolicies', 'GET', query_params={'projectId': project_id, 'datasetIds': dataset_ids, 'featureGroupId': feature_group_id, 'modelIds': model_ids, 'deploymentIds': deployment_ids, 'batchPredictionIds': batch_prediction_ids, 'modelMonitorIds': model_monitor_ids, 'notebookIds': notebook_ids}, parse_type=RefreshPolicy)

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
        return self._call_api('downloadBatchPredictionResultChunk', 'GET', query_params={'batchPredictionVersion': batch_prediction_version, 'offset': offset, 'chunkSize': chunk_size}, streamable_response=True, retry_500=True)

    def get_batch_prediction_connector_errors(self, batch_prediction_version: str) -> io.BytesIO:
        """Returns a stream containing the batch prediction database connection write errors, if any writes failed for the specified batch prediction job.

        Args:
            batch_prediction_version (str): Unique string identifier of the batch prediction job to get the errors for."""
        return self._call_api('getBatchPredictionConnectorErrors', 'GET', query_params={'batchPredictionVersion': batch_prediction_version}, streamable_response=True)

    def list_batch_predictions(self, project_id: str, limit: int = None) -> List[BatchPrediction]:
        """Retrieves a list of batch predictions in the project.

        Args:
            project_id (str): Unique string identifier of the project.
            limit (int): Maximum number of batch predictions to return. We'll have internal limit if not set.

        Returns:
            list[BatchPrediction]: List of batch prediction jobs."""
        return self._call_api('listBatchPredictions', 'GET', query_params={'projectId': project_id, 'limit': limit}, parse_type=BatchPrediction)

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

    def get_batch_prediction_version_logs(self, batch_prediction_version: str) -> BatchPredictionVersionLogs:
        """Retrieves the batch prediction logs.

        Args:
            batch_prediction_version (str): The unique version ID of the batch prediction version.

        Returns:
            BatchPredictionVersionLogs: The logs for the specified batch prediction version."""
        return self._call_api('getBatchPredictionVersionLogs', 'GET', query_params={'batchPredictionVersion': batch_prediction_version}, parse_type=BatchPredictionVersionLogs)

    def get_deployment_statistics_over_time(self, deployment_id: str, start_date: str, end_date: str) -> DeploymentStatistics:
        """Return basic access statistics for the given window

        Args:
            deployment_id (str): Unique string identifier of the deployment created under the project.
            start_date (str): Timeline start date in ISO format.
            end_date (str): Timeline end date in ISO format. The date range must be 7 days or less.

        Returns:
            DeploymentStatistics: Object describing Time series data of the number of requests and latency over the specified time period."""
        return self._call_api('getDeploymentStatisticsOverTime', 'GET', query_params={'deploymentId': deployment_id, 'startDate': start_date, 'endDate': end_date}, parse_type=DeploymentStatistics)

    def get_data(self, feature_group_id: str, primary_key: str = None, num_rows: int = None) -> List[FeatureGroupRow]:
        """Gets the feature group rows for online updatable feature groups.

        If primary key is set, row corresponding to primary_key is returned.
        If num_rows is set, we return maximum of num_rows latest updated rows.


        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            primary_key (str): The primary key value for which to retrieve the feature group row (only for online feature groups).
            num_rows (int): Maximum number of rows to return from the feature group

        Returns:
            list[FeatureGroupRow]: A list of feature group rows."""
        return self._proxy_request('getData', 'GET', query_params={'featureGroupId': feature_group_id, 'primaryKey': primary_key, 'numRows': num_rows}, parse_type=FeatureGroupRow, is_sync=True)

    def describe_python_function(self, name: str) -> PythonFunction:
        """Describe a Python Function.

        Args:
            name (str): The name to identify the Python function. Must be a valid Python identifier.

        Returns:
            PythonFunction: The Python function object."""
        return self._call_api('describePythonFunction', 'GET', query_params={'name': name}, parse_type=PythonFunction)

    def list_python_functions(self, function_type: str = 'FEATURE_GROUP') -> List[PythonFunction]:
        """List all python functions within the organization.

        Args:
            function_type (str): Optional argument to specify the type of function to list Python functions for. Default is FEATURE_GROUP, but can also be PLOTLY_FIG.

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

    def describe_pipeline_step_version(self, pipeline_step_version: str) -> PipelineStepVersion:
        """Describes a pipeline step version.

        Args:
            pipeline_step_version (str): The ID of the pipeline step version.

        Returns:
            PipelineStepVersion: An object describing the pipeline step version."""
        return self._call_api('describePipelineStepVersion', 'GET', query_params={'pipelineStepVersion': pipeline_step_version}, parse_type=PipelineStepVersion)

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

    def list_algorithms(self, problem_type: Union[ProblemType, str] = None, project_id: str = None) -> List[Algorithm]:
        """List all custom algorithms, with optional filtering on Problem Type and Project ID

        Args:
            problem_type (ProblemType): The problem type to query. If `None`, return all algorithms in the organization.
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

    def get_organization_secret(self, secret_key: str) -> OrganizationSecret:
        """Gets a secret.

        Args:
            secret_key (str): The secret key.

        Returns:
            OrganizationSecret: The secret."""
        return self._call_api('getOrganizationSecret', 'GET', query_params={'secretKey': secret_key}, parse_type=OrganizationSecret)

    def list_organization_secrets(self) -> List[OrganizationSecret]:
        """Lists all secrets for an organization.

        Returns:
            list[OrganizationSecret]: list of secrets belonging to the organization."""
        return self._call_api('listOrganizationSecrets', 'GET', query_params={}, parse_type=OrganizationSecret)

    def get_app_user_group_sign_in_token(self, user_group_id: str, email: str, name: str) -> AppUserGroupSignInToken:
        """Get a token for a user group user to sign in.

        Args:
            user_group_id (str): The ID of the user group.
            email (str): The email of the user.
            name (str): The name of the user.

        Returns:
            AppUserGroupSignInToken: The token to sign in the user"""
        return self._call_api('getAppUserGroupSignInToken', 'GET', query_params={'userGroupId': user_group_id, 'email': email, 'name': name}, parse_type=AppUserGroupSignInToken)

    def query_feature_group_code_generator(self, query: str, language: str, project_id: str = None) -> LlmResponse:
        """Send a query to the feature group code generator tool to generate code for the query.

        Args:
            query (str): A natural language query which specifies what the user wants out of the feature group or its code.
            language (str): The language in which code is to be generated. One of 'sql' or 'python'.
            project_id (str): A unique string identifier of the project in context of which the query is.

        Returns:
            LlmResponse: The response from the model, raw text and parsed components."""
        return self._call_api('queryFeatureGroupCodeGenerator', 'GET', query_params={'query': query, 'language': language, 'projectId': project_id}, parse_type=LlmResponse)

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
        """Gets a chat session from Data Science Co-pilot.

        Args:
            chat_session_id (str): Unique ID of the chat session.

        Returns:
            ChatSession: The chat session with Data Science Co-pilot"""
        return self._call_api('getChatSession', 'GET', query_params={'chatSessionId': chat_session_id}, parse_type=ChatSession)

    def list_chat_sessions(self, most_recent_per_project: bool = False) -> ChatSession:
        """Lists all chat sessions for the current user

        Args:
            most_recent_per_project (bool): An optional parameter whether to only return the most recent chat session per project. Default False.

        Returns:
            ChatSession: The chat sessions with Data Science Co-pilot"""
        return self._call_api('listChatSessions', 'GET', query_params={'mostRecentPerProject': most_recent_per_project}, parse_type=ChatSession)

    def get_deployment_conversation(self, deployment_conversation_id: str = None, external_session_id: str = None, deployment_id: str = None, filter_intermediate_conversation_events: bool = True, get_unused_document_uploads: bool = False, start: int = None, limit: int = None) -> DeploymentConversation:
        """Gets a deployment conversation.

        Args:
            deployment_conversation_id (str): Unique ID of the conversation. One of deployment_conversation_id or external_session_id must be provided.
            external_session_id (str): External session ID of the conversation.
            deployment_id (str): The deployment this conversation belongs to. This is required if not logged in.
            filter_intermediate_conversation_events (bool): If true, intermediate conversation events will be filtered out. Default is true.
            get_unused_document_uploads (bool): If true, unused document uploads will be returned. Default is false.
            start (int): The start index of the conversation.
            limit (int): The limit of the conversation.

        Returns:
            DeploymentConversation: The deployment conversation."""
        return self._proxy_request('getDeploymentConversation', 'GET', query_params={'deploymentConversationId': deployment_conversation_id, 'externalSessionId': external_session_id, 'deploymentId': deployment_id, 'filterIntermediateConversationEvents': filter_intermediate_conversation_events, 'getUnusedDocumentUploads': get_unused_document_uploads, 'start': start, 'limit': limit}, parse_type=DeploymentConversation, is_sync=True)

    def list_deployment_conversations(self, deployment_id: str = None, external_application_id: str = None, conversation_type: Union[DeploymentConversationType, str] = None, fetch_last_llm_info: bool = False, limit: int = None, search: str = None) -> List[DeploymentConversation]:
        """Lists all conversations for the given deployment and current user.

        Args:
            deployment_id (str): The deployment to get conversations for.
            external_application_id (str): The external application id associated with the deployment conversation. If specified, only conversations created on that application will be listed.
            conversation_type (DeploymentConversationType): The type of the conversation indicating its origin.
            fetch_last_llm_info (bool): If true, the LLM info for the most recent conversation will be fetched. Only applicable for system-created bots.
            limit (int): The number of conversations to return. Defaults to 600.
            search (str): The search query to filter conversations by title.

        Returns:
            list[DeploymentConversation]: The deployment conversations."""
        return self._proxy_request('listDeploymentConversations', 'GET', query_params={'deploymentId': deployment_id, 'externalApplicationId': external_application_id, 'conversationType': conversation_type, 'fetchLastLlmInfo': fetch_last_llm_info, 'limit': limit, 'search': search}, parse_type=DeploymentConversation, is_sync=True)

    def export_deployment_conversation(self, deployment_conversation_id: str = None, external_session_id: str = None) -> DeploymentConversationExport:
        """Export a Deployment Conversation.

        Args:
            deployment_conversation_id (str): A unique string identifier associated with the deployment conversation.
            external_session_id (str): The external session id associated with the deployment conversation. One of deployment_conversation_id or external_session_id must be provided.

        Returns:
            DeploymentConversationExport: The deployment conversation html export."""
        return self._proxy_request('exportDeploymentConversation', 'GET', query_params={'deploymentConversationId': deployment_conversation_id, 'externalSessionId': external_session_id}, parse_type=DeploymentConversationExport, is_sync=True)

    def get_app_user_group(self, user_group_id: str) -> AppUserGroup:
        """Gets an App User Group.

        Args:
            user_group_id (str): The ID of the App User Group.

        Returns:
            AppUserGroup: The App User Group."""
        return self._call_api('getAppUserGroup', 'GET', query_params={'userGroupId': user_group_id}, parse_type=AppUserGroup)

    def describe_external_application(self, external_application_id: str) -> ExternalApplication:
        """Describes an External Application.

        Args:
            external_application_id (str): The ID of the External Application.

        Returns:
            ExternalApplication: The External Application."""
        return self._call_api('describeExternalApplication', 'GET', query_params={'externalApplicationId': external_application_id}, parse_type=ExternalApplication)

    def list_external_applications(self) -> List[ExternalApplication]:
        """Lists External Applications in an organization.

        Returns:
            list[ExternalApplication]: List of External Applications."""
        return self._call_api('listExternalApplications', 'GET', query_params={}, parse_type=ExternalApplication)

    def download_agent_attachment(self, deployment_id: str, attachment_id: str) -> io.BytesIO:
        """Return an agent attachment.

        Args:
            deployment_id (str): The deployment ID.
            attachment_id (str): The attachment ID."""
        return self._proxy_request('downloadAgentAttachment', 'GET', query_params={'deploymentId': deployment_id, 'attachmentId': attachment_id}, is_sync=True, streamable_response=True)

    def describe_agent(self, agent_id: str) -> Agent:
        """Retrieves a full description of the specified model.

        Args:
            agent_id (str): Unique string identifier associated with the model.

        Returns:
            Agent: Description of the agent."""
        return self._call_api('describeAgent', 'GET', query_params={'agentId': agent_id}, parse_type=Agent)

    def describe_agent_version(self, agent_version: str) -> AgentVersion:
        """Retrieves a full description of the specified agent version.

        Args:
            agent_version (str): Unique string identifier of the agent version.

        Returns:
            AgentVersion: A agent version."""
        return self._call_api('describeAgentVersion', 'GET', query_params={'agentVersion': agent_version}, parse_type=AgentVersion)

    def search_feature_groups(self, text: str, num_results: int = 10, project_id: str = None, feature_group_ids: List = None) -> List[OrganizationSearchResult]:
        """Search feature groups based on text and filters.

        Args:
            text (str): Text to use for approximately matching feature groups.
            num_results (int): The maximum number of search results to retrieve. The length of the returned list is less than or equal to num_results.
            project_id (str): The ID of the project in which to restrict the search, if specified.
            feature_group_ids (List): A list of feagure group IDs to restrict the search to.

        Returns:
            list[OrganizationSearchResult]: A list of search results, each containing the retrieved object and its relevance score"""
        return self._call_api('searchFeatureGroups', 'GET', query_params={'text': text, 'numResults': num_results, 'projectId': project_id, 'featureGroupIds': feature_group_ids}, parse_type=OrganizationSearchResult)

    def list_agents(self, project_id: str) -> List[Agent]:
        """Retrieves the list of agents in the specified project.

        Args:
            project_id (str): The unique identifier associated with the project.

        Returns:
            list[Agent]: A list of agents in the project."""
        return self._call_api('listAgents', 'GET', query_params={'projectId': project_id}, parse_type=Agent)

    def list_agent_versions(self, agent_id: str, limit: int = 100, start_after_version: str = None) -> List[AgentVersion]:
        """List all versions of an agent.

        Args:
            agent_id (str): The unique identifier associated with the agent.
            limit (int): If provided, limits the number of agent versions returned.
            start_after_version (str): Unique string identifier of the version after which the list starts.

        Returns:
            list[AgentVersion]: An array of Agent versions."""
        return self._call_api('listAgentVersions', 'GET', query_params={'agentId': agent_id, 'limit': limit, 'startAfterVersion': start_after_version}, parse_type=AgentVersion)

    def copy_agent(self, agent_id: str, project_id: str = None) -> Agent:
        """Creates a copy of the input agent

        Args:
            agent_id (str): The unique id of the agent whose copy is to be generated.
            project_id (str): Project id to create the new agent to. By default it picks up the source agent's project id.

        Returns:
            Agent: The newly generated agent."""
        return self._call_api('copyAgent', 'GET', query_params={'agentId': agent_id, 'projectId': project_id}, parse_type=Agent)

    def list_llm_apps(self) -> List[LlmApp]:
        """Lists all available LLM Apps, which are LLMs tailored to achieve a specific task like code generation for a specific service's API.

        Returns:
            list[LlmApp]: A list of LLM Apps."""
        return self._call_api('listLLMApps', 'GET', query_params={}, parse_type=LlmApp)

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
            limit (int): The number of vector store versions to retrieve. The maximum value is 100.
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
    function_source_code = get_clean_function_source_code(train_function)
    predict_function_name, predict_many_function_name, initialize_function_name = None, None, None
    if predict_function is not None:
        predict_function_name = predict_function.__name__
        function_source_code = function_source_code + '\n\n' + \
            get_clean_function_source_code(predict_function)
    if predict_many_function is not None:
        predict_many_function_name = predict_many_function.__name__
        function_source_code = function_source_code + '\n\n' + \
            get_clean_function_source_code(predict_many_function)
    if initialize_function is not None:
        initialize_function_name = initialize_function.__name__
        function_source_code = function_source_code + '\n\n' + \
            get_clean_function_source_code(initialize_function)
    if common_functions:
        for func in common_functions:
            function_source_code = function_source_code + \
                '\n\n' + get_clean_function_source_code(func)
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
    if not isinstance(included_modules, list) or not all([isinstance(m, str) for m in included_modules or []]):
        raise ValueError('Included modules must be of type list of strings')
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

    def get_assignments_online_with_new_inputs(self, deployment_token: str, deployment_id: str, assignments_df: pd.DataFrame = None, constraints_df: pd.DataFrame = None, constraint_equations_df: pd.DataFrame = None, feature_mapping_dict: dict = None, solve_time_limit_seconds: float = None, optimality_gap_limit: float = None):
        """
        Get alternative positive assignments for given query. Optimal assignments are ignored and the alternative assignments are returned instead.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it can be safely embedded in an application or website.
            deployment_id (ID): The unique identifier of a deployment created under the project.
            assignments_df (pd.DataFrame): A dataframe with all the variables involved in the optimization problem
            constraints_df (pd.DataFrame): A dataframe of individual constraints, and variables in them
            constraint_equations_df (pd.DataFrame): A dataframe which tells us about the operator / constant / penalty etc of a constraint
                                                    This gives us some data which is needed to make sense of the constraints_df.
            solve_time_limit_seconds (float): Maximum time in seconds to spend solving the query.
            optimality_gap_limit (float): Optimality gap we want to come within, after which we accept the solution as valid. (0 means we only want an optimal solution). it is abs(best_solution_found - best_bound) / abs(best_solution_found)

        Returns:
            OptimizationAssignment: The assignments for a given query.
        """

        def _serialize_df_with_dtypes(df):
            # Get dtypes dictionary
            dtypes_dict = df.dtypes.apply(lambda x: str(x)).to_dict()

            # Handle special dtypes
            for col, dtype in dtypes_dict.items():
                if 'datetime' in dtype.lower():
                    dtypes_dict[col] = 'datetime'
                elif 'category' in dtype.lower():
                    dtypes_dict[col] = 'category'

            # Convert DataFrame to JSON
            json_data = df.to_json(date_format='iso')

            # Create final dictionary with both data and dtypes
            serialized = {
                'data': json_data,
                'dtypes': dtypes_dict
            }

            return json.dumps(serialized)

        serialized_assignments_df = _serialize_df_with_dtypes(assignments_df)
        serialized_constraints_df = _serialize_df_with_dtypes(constraints_df)
        serialized_constraint_equations_df = _serialize_df_with_dtypes(
            constraint_equations_df)

        query_data = {'assignments_df': serialized_assignments_df,
                      'constraints_df': serialized_constraints_df,
                      'constraint_equations_df': serialized_constraint_equations_df,
                      'feature_mapping_dict': feature_mapping_dict}

        result = self.get_assignments_online_with_new_serialized_inputs(
            deployment_token=deployment_token, deployment_id=deployment_id, query_data=query_data, solve_time_limit_seconds=solve_time_limit_seconds, optimality_gap_limit=optimality_gap_limit)
        return result

    def get_optimization_input_dataframes_with_new_inputs(self, deployment_token: str, deployment_id: str, query_data: dict):
        """
        Get assignments for given query, with new inputs

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it can be safely embedded in an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            query_data (dict): a dictionary with various key: value pairs corresponding to various updated FGs in the FG tree, which we want to update to compute new top level FGs for online solve. values are dataframes and keys are their names. Names should be same as the ones used during training.

        Returns:
            OptimizationAssignment: The output dataframes for a given query.
        """
        def _serialize_df_with_dtypes(df):
            # Get dtypes dictionary
            dtypes_dict = df.dtypes.apply(lambda x: str(x)).to_dict()

            # Handle special dtypes
            for col, dtype in dtypes_dict.items():
                if 'datetime' in dtype.lower():
                    dtypes_dict[col] = 'datetime'
                elif 'category' in dtype.lower():
                    dtypes_dict[col] = 'category'

            # Convert DataFrame to JSON
            json_data = df.to_json(date_format='iso')

            # Create final dictionary with both data and dtypes
            serialized = {
                'data': json_data,
                'dtypes': dtypes_dict
            }

            return json.dumps(serialized)

        query_data = {name: _serialize_df_with_dtypes(
            df) for name, df in query_data.items()}

        result = self.get_optimization_inputs_from_serialized(
            deployment_token=deployment_token, deployment_id=deployment_id, query_data=query_data)
        return result

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
            df (pandas.DataFrame): The dataframe to upload and use as the data source for the feature group
            clean_column_names (bool): If true, the dataframe's column names will be automatically cleaned to be complaint with Abacus.AI's column requirements. Otherwise it will raise a ValueError.
        """
        df = self._validate_pandas_df(df, clean_column_names)
        dataset = self.create_dataset_from_pandas(
            feature_group_table_name=table_name, df=df)
        return dataset.describe_feature_group()

    def update_feature_group_from_pandas_df(self, table_name: str, df, clean_column_names: bool = False) -> FeatureGroup:
        """Updates a DATASET Feature Group from a local Pandas DataFrame.

        Args:
            table_name (str): The table name of the existing feature group to update. A feature group with this name must exist and must have source type DATASET.
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
            timeout (int): If waiting for upload, time out after this limit.
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

    def create_prediction_operator_from_functions(self, name: str, project_id: str,  predict_function: callable = None, initialize_function: callable = None, feature_group_ids: list = None, cpu_size: str = None, memory: int = None, included_modules: list = None, package_requirements: list = None, use_gpu: bool = False):
        """
        Create a new prediction operator.

        Args:
            name (str): Name of the prediction operator.
            project_id (str): The unique ID of the associated project.
            predict_function (callable): The function that will be executed to run predictions.
            initialize_function (callable): The initialization function that can generate anything used by predictions, based on input feature groups.
            feature_group_ids (list): List of feature groups that are supplied to the initialize function as parameters. Each of the parameters are materialized Dataframes.
            cpu_size (str): Size of the CPU for the prediction operator.
            memory (int): Memory (in GB) for the  prediction operator.
            included_modules (list): List of names of user-created modules that will be included, which is equivalent to 'from module import *'
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            use_gpu (bool): Whether this rediction operator needs gpu.
        Returns
            PredictionOperator: The updated prediction operator object.
        """
        if initialize_function and not predict_function:
            raise Exception(
                'please provide predict function along with the initialize function')
        function_source_code = None
        initialize_function_name = None
        predict_function_name = None
        if predict_function:
            function_source_code = get_clean_function_source_code(
                predict_function)
            predict_function_name = predict_function.__name__
        if initialize_function is not None:
            initialize_function_name = initialize_function.__name__
            function_source_code = get_clean_function_source_code(
                initialize_function) + '\n\n' + function_source_code
        if function_source_code:
            function_source_code = include_modules(
                function_source_code, included_modules)
        return self.create_prediction_operator(name=name, project_id=project_id, source_code=function_source_code, predict_function_name=predict_function_name, initialize_function_name=initialize_function_name, feature_group_ids=feature_group_ids, cpu_size=cpu_size, memory=memory, package_requirements=package_requirements, use_gpu=use_gpu)

    def update_prediction_operator_from_functions(self, prediction_operator_id: str, name: str = None, predict_function: callable = None, initialize_function: callable = None, feature_group_ids: list = None, cpu_size: str = None, memory: int = None, included_modules: list = None, package_requirements: list = None, use_gpu: bool = False):
        """
        Update an existing prediction operator.

        Args:
            prediction_operator_id (str): The unique ID of the prediction operator.
            name (str): The name of the prediction operator
            predict_function (callable): The predict function callable to serialize and upload
            initialize_function (callable): The initialize function callable to serialize and upload
            feature_group_ids (list): List of feature groups that are supplied to the initialize function as parameters. Each of the parameters are materialized Dataframes. The order should match the initialize function's parameters.
            cpu_size (str): Size of the cpu for the training function
            memory (int): Memory (in GB) for the training function
            included_modules (list): List of names of user-created modules that will be included, which is equivalent to 'from module import *'
            package_requirements (List): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            use_gpu (bool): Whether this prediction needs gpu
        """
        function_source_code = None
        initialize_function_name = None
        predict_function_name = None
        if predict_function:
            function_source_code = get_clean_function_source_code(
                predict_function)
            predict_function_name = predict_function.__name__
        if initialize_function is not None:
            initialize_function_name = initialize_function.__name__
            function_source_code = get_clean_function_source_code(
                initialize_function) + '\n\n' + function_source_code
        if function_source_code:
            function_source_code = include_modules(
                function_source_code, included_modules)
        return self.update_prediction_operator(prediction_operator_id=prediction_operator_id, name=name, source_code=function_source_code, predict_function_name=predict_function_name, initialize_function_name=initialize_function_name, feature_group_ids=feature_group_ids, cpu_size=cpu_size, memory=memory, package_requirements=package_requirements, use_gpu=use_gpu)

    def create_model_from_functions(self, project_id: str, train_function: callable, predict_function: callable = None, training_input_tables: list = None, predict_many_function: callable = None, initialize_function: callable = None, cpu_size: str = None, memory: int = None, training_config: dict = None, exclusive_run: bool = False, included_modules: list = None, package_requirements: list = None, name: str = None, use_gpu: bool = False, is_thread_safe: bool = None):
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
            training_config (TrainingConfig): Training configuration
            exclusive_run (bool): Decides if this model will be run exclusively or along with other Abacus.AI algorithms
            package_requirements (List): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            included_modules (list): List of names of user-created modules that will be included, which is equivalent to 'from module import *'
            name (str): The name of the model
            use_gpu (bool): Whether this model needs gpu
            is_thread_safe (bool): Whether the model is thread safe
        """
        function_source_code, train_function_name, predict_function_name, predict_many_function_name, initialize_function_name = get_source_code_info(
            train_function, predict_function, predict_many_function, initialize_function)
        function_source_code = include_modules(
            function_source_code, included_modules)
        return self.create_model_from_python(project_id=project_id, function_source_code=function_source_code, train_function_name=train_function_name, predict_function_name=predict_function_name, predict_many_function_name=predict_many_function_name, initialize_function_name=initialize_function_name, training_input_tables=training_input_tables, training_config=training_config, cpu_size=cpu_size, memory=memory, exclusive_run=exclusive_run, package_requirements=package_requirements, name=name, use_gpu=use_gpu, is_thread_safe=is_thread_safe)

    def update_model_from_functions(self, model_id: str, train_function: callable, predict_function: callable = None, predict_many_function: callable = None, initialize_function: callable = None, training_input_tables: list = None, cpu_size: str = None, memory: int = None, included_modules: list = None, package_requirements: list = None, use_gpu: bool = False, is_thread_safe: bool = None):
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
            included_modules (list): List of names of user-created modules that will be included, which is equivalent to 'from module import *'
            use_gpu (bool): Whether this model needs gpu
            is_thread_safe (bool): Whether the model is thread safe
        """
        function_source_code, train_function_name, predict_function_name, predict_many_function_name, initialize_function_name = get_source_code_info(
            train_function, predict_function, predict_many_function, initialize_function)
        function_source_code = include_modules(
            function_source_code, included_modules)
        return self.update_python_model(model_id=model_id, function_source_code=function_source_code, train_function_name=train_function_name, predict_function_name=predict_function_name, predict_many_function_name=predict_many_function_name, initialize_function_name=initialize_function_name, training_input_tables=training_input_tables, cpu_size=cpu_size, memory=memory, package_requirements=package_requirements, use_gpu=use_gpu, is_thread_safe=is_thread_safe)

    def create_pipeline_step_from_function(self,
                                           pipeline_id: str,
                                           step_name: str,
                                           function: callable,
                                           step_input_mappings: list = None,
                                           output_variable_mappings: list = None,
                                           step_dependencies: list = None,
                                           package_requirements: list = None,
                                           cpu_size: str = None,
                                           memory: int = None,
                                           included_modules: list = None,
                                           timeout: int = None):
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
            included_modules (list): List of names of user-created modules that will be included, which is equivalent to 'from module import *'
            timeout (int): Timeout for how long the step can run in minutes, default is 300 minutes.
        """
        source_code = get_clean_function_source_code(function)

        source_code = include_modules(source_code, included_modules)

        return self.create_pipeline_step(pipeline_id=pipeline_id,
                                         step_name=step_name,
                                         function_name=function.__name__,
                                         source_code=source_code,
                                         step_input_mappings=step_input_mappings,
                                         output_variable_mappings=output_variable_mappings,
                                         step_dependencies=step_dependencies,
                                         package_requirements=package_requirements,
                                         cpu_size=cpu_size,
                                         memory=memory,
                                         timeout=timeout)

    def update_pipeline_step_from_function(self,
                                           pipeline_step_id: str,
                                           function: callable,
                                           step_input_mappings: list = None,
                                           output_variable_mappings: list = None,
                                           step_dependencies: list = None,
                                           package_requirements: list = None,
                                           cpu_size: str = None,
                                           memory: int = None,
                                           included_modules: list = None,
                                           timeout: int = None):
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
            included_modules (list): List of names of user-created modules that will be included, which is equivalent to 'from module import *'
            timeout (int): Timeout for the step in minutes, default is 300 minutes.
        """
        source_code = get_clean_function_source_code(function)

        source_code = include_modules(source_code, included_modules)

        return self.update_pipeline_step(pipeline_step_id=pipeline_step_id,
                                         function_name=function.__name__,
                                         source_code=source_code,
                                         step_input_mappings=step_input_mappings,
                                         output_variable_mappings=output_variable_mappings,
                                         step_dependencies=step_dependencies,
                                         package_requirements=package_requirements,
                                         cpu_size=cpu_size,
                                         memory=memory,
                                         timeout=timeout)

    def create_python_function_from_function(self,
                                             name: str,
                                             function: callable,
                                             function_variable_mappings: list = None,
                                             package_requirements: list = None,
                                             function_type: str = PythonFunctionType.FEATURE_GROUP.value,
                                             description: str = None):
        """
        Creates a custom Python function

        Args:
            name (str): The name to identify the Python function.
            function (callable): The function callable to serialize and upload.
            function_variable_mappings (List<PythonFunctionArguments>): List of Python function arguments.
            package_requirements (List): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            function_type (PythonFunctionType): Type of Python function to create. Default is FEATURE_GROUP, but can also be PLOTLY_FIG.
            description (str): Description of the Python function.
        """
        function_source = None
        python_function_name = None
        if function:
            function_source = get_clean_function_source_code(function)
            python_function_name = function.__name__
        else:
            raise MissingParameterError(
                'Please provide a function to create a python function', http_status=400)
        return self.create_python_function(name=name,
                                           source_code=function_source,
                                           function_name=python_function_name,
                                           function_variable_mappings=function_variable_mappings,
                                           package_requirements=package_requirements,
                                           function_type=function_type,
                                           description=description)

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
            included_modules (list): List of names of user-created modules that will be included, which is equivalent to 'from module import *'
        """
        function_source = None
        function_name = None
        if function:
            function_source = get_clean_function_source_code(function)
            function_source = include_modules(
                function_source, included_modules)
            function_name = function.__name__
            python_function_name = python_function_name or function.__name__
        return self.create_feature_group_from_function(table_name=table_name, function_source_code=function_source, input_feature_groups=input_tables, python_function_name=python_function_name, function_name=function_name, package_requirements=package_requirements, python_function_bindings=python_function_bindings, cpu_size=cpu_size, memory=memory)

    def update_python_function_code(self, name: str, function: callable = None, function_variable_mappings: list = None, package_requirements: list = None, included_modules: list = None):
        """
        Update custom python function with user inputs for the given python function.

        Args:
            name (String): The unique name to identify the python function in an organization.
            function (callable): The function callable to serialize and upload.
            function_variable_mappings (List<PythonFunctionArguments>): List of python function arguments
            package_requirements (List): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            included_modules (list): List of names of user-created modules that will be included, which is equivalent to 'from module import *'
        Returns:
            PythonFunction: The python_function object.
        """
        source_code = get_clean_function_source_code(function)
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
            is_default_enabled (bool): Whether train with the algorithm by default
            project_id (Unique String Identifier): The unique version ID of the project
            use_gpu (Boolean): Whether this algorithm needs to run on GPU
            package_requirements (List): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            included_modules (list): List of names of user-created modules that will be included, which is equivalent to 'from module import *'
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
            included_modules (list): List of names of user-created modules that will be included, which is equivalent to 'from module import *'

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

    def get_train_function_input(self, project_id: str, training_table_names: list = None, training_data_parameter_name_override: dict = None, training_config_parameter_name_override: str = None, training_config: dict = None, custom_algorithm_config: Any = None):
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

    def get_train_function_input_from_model_version(self, model_version: str, algorithm: str = None, training_config: dict = None, custom_algorithm_config: Any = None):
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

    def run_workflow_graph(self, workflow_graph: WorkflowGraph, sample_user_inputs: dict = {}, agent_workflow_node_id: str = None, agent_interface: AgentInterface = None, package_requirements: list = None):
        """
        Validates the workflow graph by running the flow using sample user inputs for an AI Agent.

        Args:
            workflow_graph (WorkflowGraph): The workflow graph to validate.
            sample_user_inputs (dict): Contains sample values for variables of type user_input for starting node
            agent_workflow_node_id (str): Node id from which we want to run workflow
            agent_interface (AgentInterface): The interface that the agent will be deployed with.
            package_requirements (list): A list of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].

        Returns:
            dict: The output variables for every node in workflow which has executed.
        """
        graph_info = self.extract_agent_workflow_information(
            workflow_graph=workflow_graph, agent_interface=agent_interface, package_requirements=package_requirements)
        workflow_vars = get_object_from_context(
            self, _request_context, 'workflow_vars', dict) or {}
        topological_dfs_stack = get_object_from_context(
            self, _request_context, 'topological_dfs_stack', list) or []
        workflow_info = run(nodes=workflow_graph['nodes'], primary_start_node=workflow_graph['primary_start_node'], graph_info=graph_info,
                            sample_user_inputs=sample_user_inputs, agent_workflow_node_id=agent_workflow_node_id, workflow_vars=workflow_vars, topological_dfs_stack=topological_dfs_stack)
        _request_context.workflow_vars = workflow_info['workflow_vars']
        _request_context.topological_dfs_stack = workflow_info['topological_dfs_stack']
        return workflow_info['run_info']

    def execute_workflow_node(self, node: WorkflowGraphNode, inputs: dict):
        """
        Execute the workflow node given input arguments. This is to be used for testing purposes only.

        Args:
            node (WorkflowGraphNode): The workflow node to be executed.
            inputs (dict): The inputs to be passed to the node function.

        Returns:
            dict: The outputs returned by node execution.
        """
        source_code = None
        function_name = None
        if node.template_metadata:
            template_metadata = node.template_metadata
            template_name = template_metadata.get('template_name')
            template = self._call_api('_getWorkflowNodeTemplate', 'GET', query_params={
                                      'name': template_name}, parse_type=WorkflowNodeTemplate)
            if not template:
                raise Exception(f'Template {template_name} not found')
            function_name = template.function_name
            configs = template_metadata.get('configs') or {}
            template_configs = [WorkflowNodeTemplateConfig.from_dict(
                template_config) for template_config in template.template_configs]
            for template_config in template_configs:
                if template_config.name not in configs:
                    if template_config.is_required:
                        raise Exception(
                            f'Missing value for required template config {template_config.name}')
                    else:
                        configs[template_config.name] = template_config.default_value
            source_code = template.source_code.format(**configs)
        else:
            source_code = node.source_code
            function_name = node.function_name
        exec(source_code, globals())
        func = eval(function_name)
        return func(**inputs)

    def create_agent_from_function(self, project_id: str, agent_function: callable, name: str = None, memory: int = None, package_requirements: list = None, description: str = None, evaluation_feature_group_id: str = None, workflow_graph: WorkflowGraph = None):
        """
        [Deprecated]
        Creates the agent from a python function

        Args:
            project_id (str): The project to create the model in
            agent_function (callable): The agent function callable to serialize and upload
            name (str): The name of the agent
            memory (int): Memory (in GB) for hosting the agent
            package_requirements (List): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            description (str): A description of the agent.
            evaluation_feature_group_id (str): The ID of the feature group to use for evaluation.
            workflow_graph (WorkflowGraph): The workflow graph for the agent.
        """
        function_source_code = get_clean_function_source_code_for_agent(
            agent_function)
        agent_function_name = agent_function.__name__
        return self.create_agent(project_id=project_id, function_source_code=function_source_code, agent_function_name=agent_function_name, name=name, memory=memory, package_requirements=package_requirements, description=description, evaluation_feature_group_id=evaluation_feature_group_id, workflow_graph=workflow_graph)

    def update_agent_with_function(self, model_id: str, agent_function: callable, memory: int = None, package_requirements: list = None, enable_binary_input: bool = None, description: str = None, workflow_graph: WorkflowGraph = None):
        """
        [Deprecated]
        Updates the agent with a new agent function.

        Args:
            model_id (str): The unique ID associated with the AI Agent to be changed.
            agent_function (callable): The new agent function callable to serialize and upload
            memory (int): Memory (in GB) for hosting the agent
            package_requirements (List): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            enable_binary_input (bool): If True, the agent will be able to accept binary data as inputs.
            description (str): A description of the agent.
            workflow_graph (WorkflowGraph): The workflow graph for the agent.
        """
        function_source_code = get_clean_function_source_code_for_agent(
            agent_function)
        agent_function_name = agent_function.__name__
        return self.update_agent(model_id=model_id, function_source_code=function_source_code, agent_function_name=agent_function_name, memory=memory, package_requirements=package_requirements, enable_binary_input=enable_binary_input, description=description, workflow_graph=workflow_graph)

    def _attempt_deployment_sql_execution(self, sql):
        deployment_id = os.environ.get('ABACUS_DEPLOYMENT_ID')
        ttl_seconds = 120  # 2 minutes

        @lru_cache()
        def _endpoint(deployment_id: str, ttl_hash: int):
            return self._call_api('_executeFeatureGroupSqlInDeploymentParams', 'GET', query_params={'deploymentId': deployment_id}, retry_500=True)

        if deployment_id:
            endpoint = _endpoint(deployment_id, time.time() // ttl_seconds)
            if endpoint:
                import pandas as pd

                with self._request(endpoint, 'GET', query_params={'sql': sql}, stream=True, retry_500=True) as response:
                    if response.status_code == 200:
                        buf = io.BytesIO(response.content)
                        return pd.read_parquet(buf, engine='pyarrow')
                    else:
                        error_json = response.json()
                        error_message = error_json.get('error')
                        error_type = error_json.get('errorType')
                        request_id = error_json.get('requestId')
                        raise _ApiExceptionFactory.from_response(
                            error_message, response.status_code, error_type, request_id)

    def execute_feature_group_sql(self, sql, fix_query_on_error: bool = False, timeout=3600, delay=2, use_latest_version=True):
        """
        Execute a SQL query on the feature groups

        Args:
            sql (str): The SQL query to execute.
            fix_query_on_error (bool): If enabled, SQL query is auto fixed if parsing fails.
            use_latest_version (bool): If enabled, executes the query on the latest version of the feature group, and if version doesn't exist, FailedDependencyError is sent. If disabled, query is executed considering the latest feature group state irrespective of the latest version of the feature group. Defaults to True

        Returns:
            pandas.DataFrame: The result of the query.
        """
        if use_latest_version:
            deployment_sql_result = self._attempt_deployment_sql_execution(sql)
            if deployment_sql_result is not None:
                return deployment_sql_result
        execute_query = self.execute_async_feature_group_operation(
            sql, fix_query_on_error=fix_query_on_error, use_latest_version=use_latest_version)
        execute_query.wait_for_execution(timeout=timeout, delay=delay)
        return execute_query.load_as_pandas()

    def _get_agent_client_type(self):
        """
        Returns the client type for the current request context.

        Returns:
            AgentClientType: The client type for the current request context.
        """
        if self._is_async_app_caller():
            return AgentClientType.MESSAGING_APP
        elif self._is_proxy_app_caller():
            return AgentClientType.CHAT_UI
        else:
            return AgentClientType.API

    def get_agent_context_chat_history(self):
        """
        Gets a history of chat messages from the current request context. Applicable within a AIAgent
        execute function.

        Returns:
            List[AgentChatMessage]: The chat history for the current request being processed by the Agent.
        """
        from .agent_chat_message import AgentChatMessage
        return get_object_from_context(self, _request_context, 'chat_history', List[AgentChatMessage]) or []

    def set_agent_context_chat_history(self, chat_history):
        """
        Sets the history of chat messages from the current request context.

        Args:
            chat_history (List[AgentChatMessage]): The chat history associated with the current request context.
        """
        _request_context.chat_history = chat_history

    def get_agent_context_chat_history_for_llm(self):
        """
        Gets a history of chat messages from the current request context. Applicable within a AIAgent
        execute function.

        Returns:
            AgentConversation: The messages in format suitable for llm.
        """
        deployment_conversation_id = self.get_agent_context_conversation_id()
        return self.construct_agent_conversation_messages_for_llm(deployment_conversation_id)

    def get_agent_context_conversation_id(self):
        """
        Gets the deployment conversation ID from the current request context. Applicable within a AIAgent
        execute function.

        Returns:
            str: The deployment conversation ID for the current request being processed by the Agent.
        """
        return get_object_from_context(self, _request_context, 'deployment_conversation_id', str)

    def set_agent_context_conversation_id(self, conversation_id):
        """
        Sets the deployment conversation ID from the current request context.

        Args:
            conversation_id (str): The deployment conversation ID for the current request being processed by the Agent.
        """
        _request_context.deployment_conversation_id = conversation_id

    def get_agent_context_external_session_id(self):
        """
        Gets the external session ID from the current request context if it has been set with the request.
        Applicable within a AIAgent execute function.

        Returns:
            str: The external session ID for the current request being processed by the Agent.
        """
        return get_object_from_context(self, _request_context, 'external_session_id', str)

    def set_agent_context_external_session_id(self, external_session_id):
        """
        Sets the external session ID from the current request context if it has been set with the request.

        Args:
            external_session_id (str): The external session ID for the current request being processed by the Agent.
        """
        _request_context.external_session_id = external_session_id

    def get_agent_context_doc_ids(self):
        """
        Gets the document ID from the current request context if a document has been uploaded with the request.
        Applicable within a AIAgent execute function.

        Returns:
            List[str]: The document IDs the current request being processed by the Agent.
        """
        return get_object_from_context(self, _request_context, 'doc_ids', List[str])

    def set_agent_context_doc_ids(self, doc_ids):
        """
        Sets the doc_ids from the current request context.

        Args:
            doc_ids (List[str]): The doc_ids associated with the current request context.
        """
        _request_context.doc_ids = doc_ids

    def get_agent_context_doc_infos(self):
        """
        Gets the document information from the current request context if documents have been uploaded with the request.
        Applicable within a AIAgent execute function.

        Returns:
            List[dict]: The document information for the current request being processed by the Agent.
        """
        return get_object_from_context(self, _request_context, 'doc_infos', List[dict])

    def set_agent_context_doc_infos(self, doc_infos):
        """
        Sets the doc_infos in the current request context.

        Args:
            doc_infos (List[dict]): The document information associated with the current request context.
        """
        _request_context.doc_infos = doc_infos

    def get_agent_context_blob_inputs(self):
        """
        Gets the BlobInputs from the current request context if a document has been uploaded with the request.
        Applicable within a AIAgent execute function.

        Returns:
            List[BlobInput]: The BlobInput the current request being processed by the Agent.
        """
        return get_object_from_context(self, _request_context, 'blob_inputs', List[BlobInput])

    def get_agent_context_user_info(self):
        """
        Gets information about the user interacting with the agent and user action if applicable.
        Applicable within a AIAgent execute function.

        Returns:
            dict: Containing email and name of the end user.
        """
        user_info = get_object_from_context(
            self, _request_context, 'user_info', dict)
        if user_info:
            user_info['client_type'] = self._get_agent_client_type()
            user_action_label = get_object_from_context(
                self, _request_context, 'action_label', str)
            if user_action_label:
                user_info['user_action_info'] = {
                    'action_label': user_action_label}
            return user_info
        else:
            raise ValueError(
                'User information not available. Please use UI interface for this agent to work.')

    def get_runtime_config(self, key: str):
        """
        Retrieve the value of a specified configuration key from the deployment's runtime settings.
        These settings can be configured in the deployment details page in the UI.
        Currently supported for AI Agents, Custom Python Model and Prediction Operators.

        Args:
            key (str): The configuration key whose value is to be fetched.

        Returns:
            str: The value associated with the specified configuration key, or None if the key does not exist.
        """
        runtime_config = get_object_from_context(
            self, _request_context, 'deployment_runtime_config', dict) or {}
        return runtime_config.get(key, None)

    def get_request_user_info(self):
        """
        Gets the user information for the current request context.

        Returns:
            dict: Containing email and name of the end user.
        """
        user_info = get_object_from_context(
            self, _request_context, 'user_info', dict)
        if user_info:
            return user_info
        else:
            raise ValueError('User information not available')

    def clear_agent_context(self):
        """
        Clears the current request context.
        """
        if hasattr(_request_context):
            _request_context.clear()

    def execute_chatllm_computer_streaming(self, computer_id: str, prompt: str, is_transient: bool = False):
        """
        Executes a prompt on a remote computer and streams computer responses to the external chat UI in real-time. Must be called from agent execution context only.

        Args:
            computer_id (str): The ID of the computer to use for the agent.
            prompt (str): The prompt to do tasks on the computer.
            is_transient (bool): If True, the message will be marked as transient and will not be persisted on reload in external chatllm UI. Transient messages are useful for streaming interim updates or results.

        Returns:
            text (str): The text responses from the computer.
        """
        request_id = self._get_agent_app_request_id()
        caller = self._get_agent_caller()
        proxy_caller = self._is_proxy_app_caller()

        if not request_id or not caller:
            raise Exception(
                'This function can only be called from within an agent execution context')

        if not caller.endswith('/'):
            caller = caller + '/'

        if proxy_caller:
            api_endpoint = f'{caller}_executeChatLLMComputerStreaming'
        else:
            raise Exception(
                'This function can only be called from within an agent execution context')

        extra_args = {'stream_type': StreamType.MESSAGE.value,
                      'response_version': '1.0', 'is_transient': is_transient}
        if hasattr(_request_context, 'agent_workflow_node_id'):
            extra_args.update(
                {'agent_workflow_node_id': _request_context.agent_workflow_node_id})

        computer_use_args = {
            'computerId': computer_id,
            'prompt': prompt
        }

        body = {
            'requestId': request_id,
            'computerUseArgs': computer_use_args,
            'extraArgs': extra_args,
        }
        body['connectionId'] = uuid4().hex

        headers = {'APIKEY': self.api_key}
        self._clean_api_objects(body)
        for _ in range(3):
            response = self._request(
                api_endpoint, method='POST', body=body, headers=headers)
            if response.status_code == 200:
                return StreamingHandler(response.json(), _request_context, is_transient=is_transient)
            elif response.status_code in (502, 503, 504):
                continue
            else:
                break
        raise Exception(
            f'Error calling ChatLLM computer streaming endpoint. Status code: {response.status_code}. Response: {response.text}')

    def streaming_evaluate_prompt(self, prompt: str = None, system_message: str = None, llm_name: Union[LLMName, str] = None, max_tokens: int = None, temperature: float = 0.0, messages: list = None, response_type: str = None, json_response_schema: dict = None, section_key: str = None):
        """
        Generate response to the prompt using the specified model. This works the same as `evaluate_prompt` but would stream the text to the UI section while generating and returns the streamed text as an object of a `str` subclass.

        Args:
            prompt (str): Prompt to use for generation.
            system_message (str): System prompt for models that support it.
            llm_name (LLMName): Name of the underlying LLM to be used for generation. Default is auto selection.
            max_tokens (int): Maximum number of tokens to generate. If set, the model will just stop generating after this token limit is reached.
            temperature (float): Temperature to use for generation. Higher temperature makes more non-deterministic responses, a value of zero makes mostly deterministic reponses. Default is 0.0. A range of 0.0 - 2.0 is allowed.
            messages (list): A list of messages to use as conversation history. For completion models like OPENAI_GPT3_5_TEXT and PALM_TEXT this should not be set. A message is a dict with attributes: is_user (bool): Whether the message is from the user. text (str): The message's text.
            response_type (str): Specifies the type of response to request from the LLM. One of 'text' and 'json'. If set to 'json', the LLM will respond with a json formatted string whose schema can be specified `json_response_schema`. Defaults to 'text'
            json_response_schema (dict): A dictionary specifying the keys/schema/parameters which LLM should adhere to in its response when `response_type` is 'json'. Each parameter is mapped to a dict with the following info - type (str) (required): Data type of the parameter description (str) (required): Description of the parameter is_required (bool) (optional): Whether the parameter is required or not.     Example:     json_response_schema={         'title': {'type': 'string', 'description': 'Article title', 'is_required': true},         'body': {'type': 'string', 'description': 'Article body'},     }
            section_key (str): Key to identify output schema section.

        Returns:
            text (str): The response from the model.
        """
        caller = self._get_agent_caller()
        request_id = self._get_agent_app_request_id()
        if prompt and not isinstance(prompt, str):
            raise ValueError('prompt must be a string')
        if system_message and not isinstance(system_message, str):
            raise ValueError('system_message must be a string')

        if caller and request_id:
            is_agent = get_object_from_context(
                self, _request_context, 'is_agent', bool)
            is_agent_api = get_object_from_context(
                self, _request_context, 'is_agent_api', bool)

            result = self._stream_llm_call(prompt=prompt, system_message=system_message, llm_name=llm_name, max_tokens=max_tokens, temperature=temperature, messages=messages,
                                           response_type=response_type, json_response_schema=json_response_schema, section_key=section_key, is_agent=is_agent, is_agent_api=is_agent_api)
        else:
            result = self.evaluate_prompt(prompt, system_message=system_message, llm_name=llm_name, max_tokens=max_tokens,
                                          temperature=temperature, messages=messages, response_type=response_type, json_response_schema=json_response_schema).content
        return StreamingHandler(result, _request_context, section_key=section_key)

    def execute_python(self, source_code: str):
        """
        Executes the given source code.

        Args:
            source_code (str): The source code to execute.

        Returns:
            stdout, stderr, exception for source_code execution
        """

        custom_globals = {}
        try:
            from abacus_internal.cloud_copy import original_builtin_print
            custom_globals['print'] = original_builtin_print
        except ImportError:
            pass

        from contextlib import redirect_stderr, redirect_stdout
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        exec_exception = None

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            try:
                # Execute the source code
                exec(source_code, custom_globals)
            except Exception as e:
                exec_exception = e

        # Retrieve the captured output
        exec_stdout = stdout_capture.getvalue()
        exec_stderr = stderr_capture.getvalue()
        return exec_stdout, exec_stderr, exec_exception

    def _get_agent_app_request_id(self):
        """
        Gets the current request ID for the current request context of async app. Applicable within a AIAgent execute function.

        Returns:
            str: The request ID for the current request being processed by the Agent.
        """
        return get_object_from_context(self, _request_context, 'request_id', str)

    def _get_agent_caller(self):
        """
        Gets the caller for the current request context. Applicable within a AIAgent execute function.

        Returns:
            str: The caller for the current request being processed by the Agent.
        """
        return get_object_from_context(self, _request_context, 'async_app_caller', str) or get_object_from_context(self, _request_context, 'proxy_app_caller', str)

    def _is_proxy_app_caller(self):
        """
        Checks if the caller is cluster-proxy app.

        Returns:
            bool: True if the caller is cluster-proxy app.
        """
        return get_object_from_context(self, _request_context, 'proxy_app_caller', str) is not None

    def _is_async_app_caller(self):
        """
        Checks if the caller is async app.

        Returns:
            bool: True if the caller is async app.
        """
        return get_object_from_context(self, _request_context, 'async_app_caller', str) is not None

    def stream_message(self, message: str, is_transient: bool = False) -> None:
        """
        Streams a message to the current request context. Applicable within a AIAgent execute function.
        If the request is from the abacus.ai app, the response will be streamed to the UI. otherwise would be logged info if used from notebook or python script.

        Args:
            message (str): The message to be streamed.
            is_transient (bool): If True, the message will be marked as transient and will not be persisted on reload in external chatllm UI. Transient messages are useful for streaming interim updates or results.
        """
        request_id = self._get_agent_app_request_id()
        caller = self._get_agent_caller()
        proxy_caller = self._is_proxy_app_caller()
        if request_id and caller:
            extra_args = {'stream_type': StreamType.MESSAGE.value,
                          'response_version': '1.0', 'is_transient': is_transient}
            if hasattr(_request_context, 'agent_workflow_node_id'):
                extra_args.update(
                    {'agent_workflow_node_id': _request_context.agent_workflow_node_id})
            self._call_aiagent_app_send_message(
                request_id, caller, message=message, extra_args=extra_args, proxy_caller=proxy_caller)
        return StreamingHandler(message, _request_context, is_transient=is_transient)

    def stream_section_output(self, section_key: str, value: str) -> None:
        """
        Streams value corresponding to a particular section to the current request context. Applicable within a AIAgent execute function.
        If the request is from the abacus.ai app, the response will be streamed to the UI. otherwise would be logged info if used from notebook or python script.

        Args:
            section_key (str): The section key to which the output corresponds.
            value (Any): The output contents.
        """
        request_id = self._get_agent_app_request_id()
        caller = self._get_agent_caller()
        proxy_caller = self._is_proxy_app_caller()
        if _is_json_serializable(value):
            message_args = {'id': section_key,
                            'type': 'text', 'mime_type': 'text/plain'}
        else:
            raise ValueError('The value is not json serializable')
        if request_id and caller:
            extra_args = {
                'stream_type': StreamType.SECTION_OUTPUT.value, 'response_version': '2.0'}
            if hasattr(_request_context, 'agent_workflow_node_id'):
                extra_args.update(
                    {'agent_workflow_node_id': _request_context.agent_workflow_node_id})
            self._call_aiagent_app_send_message(
                request_id, caller, message=value, message_args=message_args, extra_args=extra_args, proxy_caller=proxy_caller)
        return StreamingHandler(value, _request_context, section_key=section_key)

    def stream_response_section(self, response_section: ResponseSection):
        """
        Streams a response section to the current request context. Applicable within a AIAgent execute function.
        If the request is from the abacus.ai app, the response will be streamed to the UI. otherwise returned as part of response if used from notebook or python script.

        Args:
            response_section (ResponseSection): The response section to be streamed.
        """
        request_id = self._get_agent_app_request_id()
        caller = self._get_agent_caller()
        proxy_caller = self._is_proxy_app_caller()
        if request_id and caller:
            segment = response_section.to_dict()
            extra_args = {'stream_type': StreamType.SEGMENT.value}
            if hasattr(_request_context, 'agent_workflow_node_id'):
                extra_args.update(
                    {'agent_workflow_node_id': _request_context.agent_workflow_node_id})
            self._call_aiagent_app_send_message(
                request_id, caller, segment=segment, extra_args=extra_args, proxy_caller=proxy_caller)
        return StreamingHandler(segment, _request_context, data_type='segment')

    def _stream_llm_call(self, section_key=None, **kwargs):
        request_id = self._get_agent_app_request_id()
        caller = self._get_agent_caller()
        proxy_caller = self._is_proxy_app_caller()
        if not request_id or not caller:
            logging.info('Please use evaluate_prompt for local testing.')
            return
        message_args = {}
        extra_args = {'stream_type': StreamType.MESSAGE.value,
                      'response_version': '1.0'}
        if section_key:
            extra_args = {
                'stream_type': StreamType.SECTION_OUTPUT.value, 'response_version': '2.0'}
            message_args = {'id': section_key,
                            'type': 'text', 'mime_type': 'text/plain'}
        if hasattr(_request_context, 'agent_workflow_node_id'):
            extra_args.update(
                {'agent_workflow_node_id': _request_context.agent_workflow_node_id})
        return self._call_aiagent_app_send_message(request_id, caller, llm_args=kwargs, message_args=message_args, extra_args=extra_args, proxy_caller=proxy_caller)

    def _call_aiagent_app_send_message(self, request_id, caller, message=None, segment=None, llm_args=None, message_args=None, extra_args=None, proxy_caller=False):
        """
        Calls the AI Agent app send message endpoint.

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
        if proxy_caller:
            api_endpont = f'{caller}_agentConversationStreamMessage'
        else:
            api_endpont = f'{caller}sendSyncMessageExecuteAgentRequest'
        body = {'requestId': request_id}
        if message:
            body['message'] = message
        elif segment:
            body['segment'] = segment
        elif llm_args:
            body['llmArgs'] = llm_args
        else:
            raise Exception('Either message or llm_args must be provided.')
        if message_args:
            body['messageArgs'] = message_args
        if extra_args:
            body['extraArgs'] = extra_args
        headers = {'APIKEY': self.api_key}
        body['connectionId'] = uuid4().hex
        self._clean_api_objects(body)
        for _ in range(3):
            response = self._request(
                api_endpont, method='POST', body=body, headers=headers)
            if response.status_code == 200:
                return response.json()
            elif response.status_code in (502, 503, 504):
                continue
        raise Exception(
            f'Error calling AI Agent app message endpoint. Status code: {response.status_code}. Response: {response.text}')

    def _status_poll(self, url: str, wait_states: set, method: str, body: dict = {}, headers: dict = None, delay: int = 1, timeout: int = 1200):
        start_time = time.time()
        while time.time() - start_time <= timeout:
            for _ in range(3):
                response = self._request(
                    url=url, method=method, body=body, headers=headers)
                if response.status_code < 500:
                    break
                time.sleep(0.5)
            response_json = response.json()
            if response_json['status'] not in wait_states:
                return response_json, response.status_code
            time.sleep(delay)
        raise Exception('Maximum timeout Exceeded')

    def execute_data_query_using_llm(self, query: str, feature_group_ids: List[str], prompt_context: str = None, llm_name: str = None,
                                     temperature: float = None, preview: bool = False, schema_document_retriever_ids: List[str] = None,
                                     timeout=3600, delay=2, use_latest_version=True):
        """
        Execute a data query using a large language model.

        Args:
            query (str): The natural language query to execute. The query is converted to a SQL query using the language model.
            feature_group_ids (List[str]): A list of feature group IDs that the query should be executed against.
            prompt_context (str): The context message used to construct the prompt for the language model. If not provide, a default context message is used.
            llm_name (str): The name of the language model to use. If not provided, the default language model is used.
            temperature (float): The temperature to use for the language model if supported. If not provided, the default temperature is used.
            preview (bool): If True, a preview of the query execution is returned.
            schema_document_retriever_ids (List[str]): A list of document retrievers to retrieve schema information for the data query. Otherwise, they are retrieved from the feature group metadata.
            timeout (int): Time limit for the call.
            delay (int): Polling interval for checking timeout.
            use_latest_version (bool): If enabled, executes the query on the latest version of the feature group, and if version doesn't exist, FailedDependencyError is sent. If disabled, query is executed considering the latest feature group state irrespective of the latest version of the feature group. Defaults to True.

        Returns:
            LlmExecutionResult: The result of the query execution. Execution results could be loaded as pandas using 'load_as_pandas', i.e., result.execution.load_as_pandas().
        """
        code = self.generate_code_for_data_query_using_llm(
            query=query, feature_group_ids=feature_group_ids, prompt_context=prompt_context, llm_name=llm_name, temperature=temperature, sql_dialect='Spark')
        if 'SELECT' not in (code.sql or ''):
            result_dict = {
                'error': 'Unable to generate SQL given current context and tables. Please clarify your prompt to generate a query'}

        result_dict = {'preview': {'sql': code.sql}}
        if not preview:
            execute_query = self.execute_async_feature_group_operation(
                code.sql, use_latest_version=use_latest_version)
            execute_query.wait_for_execution(timeout=timeout, delay=delay)
            execute_query_dict = {'error': execute_query.error} if execute_query.error else {
                'featureGroupOperationRunId': execute_query.feature_group_operation_run_id,
                'status': execute_query.status,
            }
            result_dict.update({'execution': execute_query_dict})

        return self._build_class(LlmExecutionResult, result_dict)

    def _get_doc_retriever_deployment_info(self, document_retriever_id: str):
        ttl_seconds = 300  # 5 minutes

        @lru_cache()
        def _cached_doc_retriever_deployment_info(document_retriever_id: str, ttl_hash: int):
            info = self._call_api('_getDocRetrieverDeploymentInfo', 'GET', query_params={
                                  'documentRetrieverId': document_retriever_id})
            deployment_token = info['deploymentToken']
            deployment_id = info['deploymentId']
            return deployment_token, deployment_id
        return _cached_doc_retriever_deployment_info(document_retriever_id, ttl_hash=time.time() // ttl_seconds)

    def get_matching_documents(self, document_retriever_id: str, query: str, filters: dict = None, limit: int = None, result_columns: list = None, max_words: int = None, num_retrieval_margin_words: int = None,
                               max_words_per_chunk: int = None, score_multiplier_column: str = None, min_score: float = None, required_phrases: list = None,
                               filter_clause: str = None, crowding_limits: Dict[str, int] = None,
                               include_text_search: bool = False) -> List[DocumentRetrieverLookupResult]:
        """Lookup document retrievers and return the matching documents from the document retriever deployed with given query.

        Original documents are splitted into chunks and stored in the document retriever. This lookup function will return the relevant chunks
        from the document retriever. The returned chunks could be expanded to include more words from the original documents and merged if they
        are overlapping, and permitted by the settings provided. The returned chunks are sorted by relevance.


        Args:
            document_retriever_id (str): A unique string identifier associated with the document retriever.
            query (str): The query to search for.
            filters (dict): A dictionary mapping column names to a list of values to restrict the retrieved search results.
            limit (int): If provided, will limit the number of results to the value specified.
            result_columns (list): If provided, will limit the column properties present in each result to those specified in this list.
            max_words (int): If provided, will limit the total number of words in the results to the value specified.
            num_retrieval_margin_words (int): If provided, will add this number of words from left and right of the returned chunks.
            max_words_per_chunk (int): If provided, will limit the number of words in each chunk to the value specified. If the value provided is smaller than the actual size of chunk on disk, which is determined during document retriever creation, the actual size of chunk will be used. I.e, chunks looked up from document retrievers will not be split into smaller chunks during lookup due to this setting.
            score_multiplier_column (str): If provided, will use the values in this column to modify the relevance score of the returned chunks. Values in this column must be numeric.
            min_score (float): If provided, will filter out the results with score lower than the value specified.
            required_phrases (list): If provided, each result will have at least one of the phrases.
            filter_clause (str): If provided, filter the results of the query using this sql where clause.
            crowding_limits (dict): A dictionary mapping metadata columns to the maximum number of results per unique value of the column. This is used to ensure diversity of metadata attribute values in the results. If a particular attribute value has already reached its maximum count, further results with that same attribute value will be excluded from the final result set.
            include_text_search (bool): If true, combine the ranking of results from a BM25 text search over the documents with the vector search using reciprocal rank fusion. It leverages both lexical and semantic matching for better overall results. It's particularly valuable in professional, technical, or specialized fields where both precision in terminology and understanding of context are important.
        Returns:
            list[DocumentRetrieverLookupResult]: The relevant documentation results found from the document retriever."""

        deployment_token, deployment_id = self._get_doc_retriever_deployment_info(
            document_retriever_id)
        return self.lookup_matches(deployment_token, deployment_id, query, filters, limit if limit is not None else 10, result_columns, max_words, num_retrieval_margin_words, max_words_per_chunk, score_multiplier_column, min_score, required_phrases, filter_clause, crowding_limits, include_text_search=include_text_search)

    def create_model_from_files(self, project_id: str, location: str, name: str = None, custom_artifact_filenames: dict = {}, model_config: dict = {}) -> Model:
        """Creates a new Model and returns Upload IDs for uploading the model artifacts.

        Use this in supported use cases to provide a pre-trained model and supporting artifacts to be hosted on our platform.


        Args:
            project_id (str): Unique string identifier associated with the project.
            location (str): Cloud location for the model.
            name (str): Name you want your model to have. Defaults to "<Project Name> Model".
            custom_artifact_filenames (dict): Optional mapping to specify which filename should be used for a given model artifact type.
            model_config (dict): Extra configurations that are specific to the model being created.

        Returns:
            Model: The new model which is being trained."""
        return self._call_api('createModelFromFiles', 'POST', query_params={}, body={'projectId': project_id, 'location': location, 'name': name, 'customArtifactFilenames': custom_artifact_filenames, 'modelConfig': model_config}, parse_type=Model)

    def create_model_from_local_files(self, project_id: str, name: str = None, optional_artifacts: list = None, model_config: dict = {}) -> ModelUpload:
        """Creates a new Model and returns Upload IDs for uploading the model artifacts.

        Use this in supported use cases to provide a pre-trained model and supporting artifacts to be hosted on our platform.


        Args:
            project_id (str): The unique ID associated with the project.
            name (str): The name you want your model to have. Defaults to "<Project Name> Model".
            optional_artifacts (list): A list of strings describing additional artifacts for the model. An example would be a verification file.
            model_config (dict): Extra configurations that are specific to the model being created.

        Returns:
            ModelUpload: Collection of upload IDs to upload the model artifacts."""
        return self._call_api('createModelFromLocalFiles', 'POST', query_params={}, body={'projectId': project_id, 'name': name, 'optionalArtifacts': optional_artifacts, 'modelConfig': model_config}, parse_type=ModelUpload)

    def create_model_version_from_files(self, model_id: str) -> ModelVersion:
        """Creates a new Model Version by re-importing from the paths specified when the model was created.

        Args:
            model_id (str): Unique string identifier of the model to create a new version of with the new model artifacts.

        Returns:
            ModelVersion: The updated model."""
        return self._call_api('createModelVersionFromFiles', 'POST', query_params={}, body={'modelId': model_id}, parse_type=ModelVersion)

    def create_model_version_from_local_files(self, model_id: str, optional_artifacts: list = None) -> ModelUpload:
        """Creates a new Model Version and returns Upload IDs for uploading the associated model artifacts.

        Args:
            model_id (str): Unique string identifier of the model to create a new version of with the new model artifacts.
            optional_artifacts (list): List of strings describing additional artifacts for the model, e.g. a verification file.

        Returns:
            ModelUpload: Collection of upload IDs to upload the model artifacts."""
        return self._call_api('createModelVersionFromLocalFiles', 'POST', query_params={}, body={'modelId': model_id, 'optionalArtifacts': optional_artifacts}, parse_type=ModelUpload)

    def get_streaming_chat_response(self, deployment_token: str, deployment_id: str, messages: list, llm_name: str = None, num_completion_tokens: int = None, system_message: str = None, temperature: float = 0.0, filter_key_values: dict = None, search_score_cutoff: float = None, chat_config: dict = None, ignore_documents: bool = False, include_search_results: bool = False):
        """Return an asynchronous generator which continues the conversation based on the input messages and search results.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            messages (list): A list of chronologically ordered messages, starting with a user message and alternating sources. A message is a dict with attributes:     is_user (bool): Whether the message is from the user.      text (str): The message's text.
            llm_name (str): Name of the specific LLM backend to use to power the chat experience
            num_completion_tokens (int): Default for maximum number of tokens for chat answers
            system_message (str): The generative LLM system message
            temperature (float): The generative LLM temperature
            filter_key_values (dict): A dictionary mapping column names to a list of values to restrict the retrieved search results.
            search_score_cutoff (float): Cutoff for the document retriever score. Matching search results below this score will be ignored.
            chat_config (dict): A dictionary specifying the query chat config override.
            ignore_documents (bool): If True, will ignore any documents and search results, and only use the messages to generate a response.
            include_search_results (bool): If True, will also return search results, if relevant. """
        headers = {'APIKEY': self.api_key}
        body = {
            'deploymentToken': deployment_token,
            'deploymentId': deployment_id,
            'messages': messages,
            'llmName': llm_name,
            'numCompletionTokens': num_completion_tokens,
            'systemMessage': system_message,
            'temperature': temperature,
            'filterKeyValues': filter_key_values,
            'searchScoreCutoff': search_score_cutoff,
            'chatConfig': chat_config,
            'ignoreDocuments': ignore_documents,
            'includeSearchResults': include_search_results
        }
        endpoint = self._get_proxy_endpoint(deployment_id, deployment_token)
        if endpoint is None:
            raise Exception(
                'API not supported, Please contact Abacus.ai support')
        return sse_asynchronous_generator(f'{endpoint}/api/getStreamingChatResponse', headers, body)

    def get_streaming_conversation_response(self, deployment_token: str, deployment_id: str, message: str, deployment_conversation_id: str = None, external_session_id: str = None, llm_name: str = None, num_completion_tokens: int = None, system_message: str = None, temperature: float = 0.0, filter_key_values: dict = None, search_score_cutoff: float = None, chat_config: dict = None, ignore_documents: bool = False, include_search_results: bool = False):
        """Return an asynchronous generator which continues the conversation based on the input messages and search results.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            message (str): A message from the user
            deployment_conversation_id (str): The unique identifier of a deployment conversation to continue. If not specified, a new one will be created.
            external_session_id (str): The user supplied unique identifier of a deployment conversation to continue. If specified, we will use this instead of a internal deployment conversation id.
            llm_name (str): Name of the specific LLM backend to use to power the chat experience
            num_completion_tokens (int): Default for maximum number of tokens for chat answers
            system_message (str): The generative LLM system message
            temperature (float): The generative LLM temperature
            filter_key_values (dict): A dictionary mapping column names to a list of values to restrict the retrieved search results.
            search_score_cutoff (float): Cutoff for the document retriever score. Matching search results below this score will be ignored.
            chat_config (dict): A dictionary specifying the query chat config override.
            ignore_documents (bool): If True, will ignore any documents and search results, and only use the messages to generate a response.
            include_search_results (bool): If True, will also return search results, if relevant. """
        headers = {'APIKEY': self.api_key}
        body = {
            'deploymentToken': deployment_token,
            'deploymentId': deployment_id,
            'message': message,
            'deploymentConversationId': deployment_conversation_id,
            'externalSessionId': external_session_id,
            'llmName': llm_name,
            'numCompletionTokens': num_completion_tokens,
            'systemMessage': system_message,
            'temperature': temperature,
            'filterKeyValues': filter_key_values,
            'searchScoreCutoff': search_score_cutoff,
            'chatConfig': chat_config,
            'ignoreDocuments': ignore_documents,
            'includeSearchResults': include_search_results
        }
        endpoint = self._get_proxy_endpoint(deployment_id, deployment_token)
        if endpoint is None:
            raise Exception(
                'API not supported, Please contact Abacus.ai support')
        return sse_asynchronous_generator(f'{endpoint}/api/getStreamingConversationResponse', headers, body)

    def execute_conversation_agent_streaming(self, deployment_token: str, deployment_id: str, arguments: list = None, keyword_arguments: dict = None, deployment_conversation_id: str = None, external_session_id: str = None,
                                             regenerate: bool = False, doc_infos: list = None, agent_workflow_node_id: str = None):
        """Return an asynchronous generator which gives out the agent response stream.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            arguments (list): A list of arguments to pass to the agent.
            keyword_arguments (dict): A dictionary of keyword arguments to pass to the agent.
            deployment_conversation_id (str): The unique identifier of a deployment conversation to continue. If not specified, a new one will be created.
            external_session_id (str): The user supplied unique identifier of a deployment conversation to continue. If specified, we will use this instead of a internal deployment conversation id.
            regenerate (bool): If True, will regenerate the conversation from the start.
            doc_infos (list): A list of dictionaries containing information about the documents uploaded with the request.
            agent_workflow_node_id (str): The unique identifier of the agent workflow node to trigger. If not specified, the primary node will be used.
        """
        headers = {'APIKEY': self.api_key}
        body = {
            'deploymentToken': deployment_token,
            'deploymentId': deployment_id,
            'arguments': arguments,
            'keywordArguments': keyword_arguments,
            'deploymentConversationId': deployment_conversation_id,
            'externalSessionId': external_session_id,
            'regenerate': regenerate,
            'docInfos': doc_infos,
            'agentWorkflowNodeId': agent_workflow_node_id
        }
        endpoint = self._get_proxy_endpoint(deployment_id, deployment_token)
        if endpoint is None:
            raise Exception(
                'API not supported, Please contact Abacus.ai support')
        return sse_asynchronous_generator(f'{endpoint}/api/executeConversationAgentStreaming', headers, body)

    def set_cache_scope(self, scope: str):
        """
        Set the scope of the cache, for example, deployment id.

        Args:
            scope (String): The key of the cache entry.
        Returns:
            None
        """
        self.cache_scope = scope

    def clear_cache_scope(self):
        """
        Clear the scope set before, and let it to automatically figure out the scope to use. If nothing found, will run in local.
        """
        self.cache_scope = None

    def set_scoped_cache_value(self, key: str, value: str, expiration_time: int = 21600):
        """
        Set the value to key in the cache scope. Scope will be automatically figured out inside a deployment, or set with set_cache_scope.
        If no scope found, will run in local.

        Args:
            key (String): The key of the cache entry.
            value (String): The value of the cache entry. Only string, integer and float numbers are supported now.
            expiration_time (int): How long to keep the cache key before expire, in seconds. Default is 6h.
        Returns:
            None
        Raises:
            InvalidParameterError: If key, value or expiration_time is invalid.
        """
        scope = self.cache_scope or os.getenv(
            'ABACUS_EXEC_SERVICE_DEPLOYMENT_ID')
        if scope:
            return self._proxy_request('_setScopedCacheValue', 'POST', body={'key': key, 'value': value, 'scope': scope, 'expirationTime': expiration_time}, is_sync=True)
        else:
            warnings.warn(
                'Using local cache as no deployment id set, expected for non-deployment environment.')
            self._cache[key] = value

    def get_scoped_cache_value(self, key: str):
        """
        Get the value of the key in the cache scope. Scope will be automatically figured out inside a deployment, or set with set_cache_scope.
        If no scope found, will run in local.

        Args:
            key (String): The key of the cache entry.
        Returns:
            value (String): The value of the key
        Raises:
            Generic404Error: if the key doesn't exist.
        """
        scope = self.cache_scope or os.getenv(
            'ABACUS_EXEC_SERVICE_DEPLOYMENT_ID')
        if scope:
            return self._proxy_request('_getScopedCacheValue', 'GET', query_params={'key': key, 'scope': scope}, is_sync=True)
        else:
            warnings.warn(
                'Using local cache as no deployment id set, expected for non-deployment environment.')
            if key not in self._cache:
                raise Generic404Error(f'Cache key {key} does not exist', 404)
            return self._cache.get(key)

    def delete_scoped_cache_key(self, key: str):
        """
        Delete the value of the key in the cache scope. Scope will be automatically figured out inside a deployment, or set with set_cache_scope.
        If no scope found, will run in local.

        Args:
            key (String): The key of the cache entry.
        Returns:
            None
        """
        scope = self.cache_scope or os.getenv(
            'ABACUS_EXEC_SERVICE_DEPLOYMENT_ID')
        if scope:
            return self._proxy_request('_deleteScopedCacheKey', 'POST', query_params={'key': key, 'scope': scope}, is_sync=True)
        else:
            warnings.warn(
                'Using local cache as no deployment id set, expected for non-deployment environment.')
            self._cache.pop(key, None)

    def set_agent_response_document_sources(self, response_document_sources: List[DocumentRetrieverLookupResult]):
        """
        Sets the document sources to be shown with the response on the conversation UI.

        Args:
            response_document_sources (List): List of document retriever results to be displayed in order.
        Returns:
            None
        """
        if hasattr(_request_context, 'response_document_sources'):
            raise Exception('Document sources cannot be set more than once')
        _request_context.agent_response_document_sources = [{
            'score': float(response_document_source.score),
            'answer': response_document_source.document,
            'source': response_document_source.document_source,
            'image_ids': response_document_source.image_ids,
            'pages': response_document_source.pages,
            'bounding_boxes': response_document_source.bounding_boxes,
            'metadata': response_document_source.metadata
        } for response_document_source in response_document_sources]

    def get_initialized_data(self):
        """
            Returns the object returned by the initialize_function during agent creation.
            Returns:
                Object returned in the initialize_function.
        """
        initialized_object = None
        if hasattr(_request_context, 'model_object'):
            initialized_object = _request_context.model_object
        return initialized_object

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

    def send_email(self, email: str, subject: str, body: str, is_html: bool = False, attachments: None = None):
        """Send an email to the specified email address with provided subject and contents.

        Args:
            email (str): The email address to send the email to.
            subject (str): The subject of the email.
            body (str): The body of the email.
            is_html (bool): Whether the body is html or not.
            attachments (None): A dictionary where the key is the filename (including the file extension), and the value is either a file-like object (e.g., an open file in binary mode) or raw file data (e.g., bytes)."""
        return self._call_api('sendEmail', 'POST', query_params={}, data={'email': json.dumps(email) if (email is not None and not isinstance(email, str)) else email, 'subject': json.dumps(subject) if (subject is not None and not isinstance(subject, str)) else subject, 'body': json.dumps(body) if (body is not None and not isinstance(body, str)) else body, 'isHtml': json.dumps(is_html) if (is_html is not None and not isinstance(is_html, str)) else is_html}, files=attachments)

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
            use_case (str): The use case that the project solves. Refer to our [guide on use cases](https://api.abacus.ai/app/help/useCases) for further details of each use case. The following enums are currently available for you to choose from:  LANGUAGE_DETECTION,  NLP_SENTIMENT,  NLP_SEARCH,  NLP_CHAT,  CHAT_LLM,  NLP_SENTENCE_BOUNDARY_DETECTION,  NLP_CLASSIFICATION,  NLP_SUMMARIZATION,  NLP_DOCUMENT_VISUALIZATION,  AI_AGENT,  EMBEDDINGS_ONLY,  MODEL_WITH_EMBEDDINGS,  TORCH_MODEL,  TORCH_MODEL_WITH_EMBEDDINGS,  PYTHON_MODEL,  NOTEBOOK_PYTHON_MODEL,  DOCKER_MODEL,  DOCKER_MODEL_WITH_EMBEDDINGS,  CUSTOMER_CHURN,  ENERGY,  EVENT_ANOMALY_DETECTION,  FINANCIAL_METRICS,  CUMULATIVE_FORECASTING,  FRAUD_ACCOUNT,  FRAUD_TRANSACTIONS,  CLOUD_SPEND,  TIMESERIES_ANOMALY,  OPERATIONS_MAINTENANCE,  PERS_PROMOTIONS,  PREDICTING,  FEATURE_STORE,  RETAIL,  SALES_FORECASTING,  SALES_SCORING,  FEED_RECOMMEND,  USER_RANKINGS,  NAMED_ENTITY_RECOGNITION,  USER_RECOMMENDATIONS,  USER_RELATED,  VISION,  VISION_REGRESSION,  VISION_OBJECT_DETECTION,  FEATURE_DRIFT,  SCHEDULING,  GENERIC_FORECASTING,  PRETRAINED_IMAGE_TEXT_DESCRIPTION,  PRETRAINED_SPEECH_RECOGNITION,  PRETRAINED_STYLE_TRANSFER,  PRETRAINED_TEXT_TO_IMAGE_GENERATION,  PRETRAINED_OCR_DOCUMENT_TO_TEXT,  THEME_ANALYSIS,  CLUSTERING,  CLUSTERING_TIMESERIES,  FINETUNED_LLM,  PRETRAINED_INSTRUCT_PIX2PIX,  PRETRAINED_TEXT_CLASSIFICATION.

        Returns:
            Project: This object represents the newly created project."""
        return self._call_api('createProject', 'POST', query_params={}, body={'name': name, 'useCase': use_case}, parse_type=Project)

    def rename_project(self, project_id: str, name: str):
        """This method renames a project after it is created.

        Args:
            project_id (str): The unique identifier for the project.
            name (str): The new name for the project."""
        return self._call_api('renameProject', 'PATCH', query_params={}, body={'projectId': project_id, 'name': name})

    def delete_project(self, project_id: str, force_delete: bool = False):
        """Delete a specified project from your organization.

        This method deletes the project, its associated trained models, and deployments. The datasets attached to the specified project remain available for use with other projects in the organization.

        This method will not delete a project that contains active deployments. Ensure that all active deployments are stopped before using the delete option.

        Note: All projects, models, and deployments cannot be recovered once they are deleted.


        Args:
            project_id (str): The unique ID of the project to delete.
            force_delete (bool): If True, the project will be deleted even if it has active deployments."""
        return self._call_api('deleteProject', 'DELETE', query_params={'projectId': project_id, 'forceDelete': force_delete})

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

    def set_project_feature_group_config(self, feature_group_id: str, project_id: str, project_config: Union[dict, ProjectFeatureGroupConfig] = None):
        """Sets a feature group's project config

        Args:
            feature_group_id (str): Unique string identifier for the feature group.
            project_id (str): Unique string identifier for the project.
            project_config (ProjectFeatureGroupConfig): Feature group's project configuration."""
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

    def add_annotation(self, annotation: dict, feature_group_id: str, feature_name: str, doc_id: str = None, feature_group_row_identifier: str = None, annotation_source: str = 'ui', status: str = None, comments: dict = None, project_id: str = None, save_metadata: bool = False, pages: list = None) -> AnnotationEntry:
        """Add an annotation entry to the database.

        Args:
            annotation (dict): The annotation to add. Format of the annotation is determined by its annotation type.
            feature_group_id (str): The ID of the feature group the annotation is on.
            feature_name (str): The name of the feature the annotation is on.
            doc_id (str): The ID of the primary document the annotation is on. At least one of the doc_id or feature_group_row_identifier must be provided in order to identify the correct annotation.
            feature_group_row_identifier (str): The key value of the feature group row the annotation is on (cast to string). Usually the feature group's primary / identifier key value. At least one of the doc_id or feature_group_row_identifier must be provided in order to identify the correct annotation.
            annotation_source (str): Indicator of whether the annotation came from the UI, bulk upload, etc.
            status (str): The status of the annotation. Can be one of 'todo', 'in_progress', 'done'. This is optional.
            comments (dict): Comments for the annotation. This is a dictionary of feature name to the corresponding comment. This is optional.
            project_id (str): The ID of the project that the annotation is associated with. This is optional.
            save_metadata (bool): Whether to save the metadata for the annotation. This is optional.
            pages (list): pages (list): List of page numbers to consider while processing the annotation. This is optional. doc_id must be provided if pages is provided.

        Returns:
            AnnotationEntry: The annotation entry that was added."""
        return self._proxy_request('addAnnotation', 'POST', query_params={}, body={'annotation': annotation, 'featureGroupId': feature_group_id, 'featureName': feature_name, 'docId': doc_id, 'featureGroupRowIdentifier': feature_group_row_identifier, 'annotationSource': annotation_source, 'status': status, 'comments': comments, 'projectId': project_id, 'saveMetadata': save_metadata, 'pages': pages}, parse_type=AnnotationEntry, is_sync=True)

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

    def get_document_to_annotate(self, feature_group_id: str, project_id: str, feature_name: str, feature_group_row_identifier: str = None, get_previous: bool = False) -> AnnotationDocument:
        """Get an available document that needs to be annotated for a annotation feature group.

        Args:
            feature_group_id (str): The ID of the feature group the annotation is on.
            project_id (str): The ID of the project that the annotation is associated with.
            feature_name (str): The name of the feature the annotation is on.
            feature_group_row_identifier (str): The key value of the feature group row the annotation is on (cast to string). Usually the primary key value. If provided, fetch the immediate next (or previous) available document.
            get_previous (bool): If True, get the previous document instead of the next document. Applicable if feature_group_row_identifier is provided.

        Returns:
            AnnotationDocument: The document to annotate."""
        return self._call_api('getDocumentToAnnotate', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'projectId': project_id, 'featureName': feature_name, 'featureGroupRowIdentifier': feature_group_row_identifier, 'getPrevious': get_previous}, parse_type=AnnotationDocument)

    def import_annotation_labels(self, feature_group_id: str, file: io.TextIOBase, annotation_type: str) -> AnnotationConfig:
        """Imports annotation labels from csv file. All valid values in the file will be imported as labels (including header row if present).

        Args:
            feature_group_id (str): The unique string identifier of the feature group.
            file (io.TextIOBase): The file to import. Must be a csv file.
            annotation_type (str): The type of the annotation.

        Returns:
            AnnotationConfig: The annotation config for the feature group."""
        return self._call_api('importAnnotationLabels', 'POST', query_params={}, data={'featureGroupId': json.dumps(feature_group_id) if (feature_group_id is not None and not isinstance(feature_group_id, str)) else feature_group_id, 'annotationType': json.dumps(annotation_type) if (annotation_type is not None and not isinstance(annotation_type, str)) else annotation_type}, parse_type=AnnotationConfig, files={'file': file})

    def create_feature_group(self, table_name: str, sql: str, description: str = None, version_limit: int = 30) -> FeatureGroup:
        """Creates a new FeatureGroup from a SQL statement.

        Args:
            table_name (str): The unique name to be given to the FeatureGroup. Can be up to 120 characters long and can only contain alphanumeric characters and underscores.
            sql (str): Input SQL statement for forming the FeatureGroup.
            description (str): The description about the FeatureGroup.
            version_limit (int): The number of versions to preserve for the FeatureGroup (minimum 30).

        Returns:
            FeatureGroup: The created FeatureGroup."""
        return self._call_api('createFeatureGroup', 'POST', query_params={}, body={'tableName': table_name, 'sql': sql, 'description': description, 'versionLimit': version_limit}, parse_type=FeatureGroup)

    def create_feature_group_from_template(self, table_name: str, feature_group_template_id: str, template_bindings: list = None, should_attach_feature_group_to_template: bool = True, description: str = None, version_limit: int = 30) -> FeatureGroup:
        """Creates a new feature group from a SQL statement.

        Args:
            table_name (str): The unique name to be given to the feature group. Can be up to 120 characters long and can only contain alphanumeric characters and underscores.
            feature_group_template_id (str): The unique ID associated with the template that will be used to create this feature group.
            template_bindings (list): Variable bindings that override the template's variable values.
            should_attach_feature_group_to_template (bool): Set to `False` to create a feature group but not leave it attached to the template that created it.
            description (str): A user-friendly description of this feature group.
            version_limit (int): The number of versions to preserve for the feature group (minimum 30).

        Returns:
            FeatureGroup: The created feature group."""
        return self._call_api('createFeatureGroupFromTemplate', 'POST', query_params={}, body={'tableName': table_name, 'featureGroupTemplateId': feature_group_template_id, 'templateBindings': template_bindings, 'shouldAttachFeatureGroupToTemplate': should_attach_feature_group_to_template, 'description': description, 'versionLimit': version_limit}, parse_type=FeatureGroup)

    def create_feature_group_from_function(self, table_name: str, function_source_code: str = None, function_name: str = None, input_feature_groups: list = None, description: str = None, cpu_size: Union[CPUSize, str] = None, memory: Union[MemorySize, str] = None, package_requirements: list = None, use_original_csv_names: bool = False, python_function_name: str = None, python_function_bindings: List = None, use_gpu: bool = None, version_limit: int = 30) -> FeatureGroup:
        """Creates a new feature in a Feature Group from user-provided code. Currently supported code languages are Python.

        If a list of input feature groups are supplied, we will provide DataFrames (pandas, in the case of Python) with the materialized feature groups for those input feature groups as arguments to the function.

        This method expects the source code to be a valid language source file containing a function. This function needs to return a DataFrame when executed; this DataFrame will be used as the materialized version of this feature group table.


        Args:
            table_name (str): The unique name to be given to the feature group. Can be up to 120 characters long and can only contain alphanumeric characters and underscores.
            function_source_code (str): Contents of a valid source code file in a supported Feature Group specification language (currently only Python). The source code should contain a function called function_name. A list of allowed import and system libraries for each language is specified in the user functions documentation section.
            function_name (str): Name of the function found in the source code that will be executed (on the optional inputs) to materialize this feature group.
            input_feature_groups (list): List of feature group names that are supplied to the function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            description (str): The description for this feature group.
            cpu_size (CPUSize): Size of the CPU for the feature group function.
            memory (MemorySize): Memory (in GB) for the feature group function.
            package_requirements (list): List of package requirements for the feature group function. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            use_original_csv_names (bool): Defaults to False, if set it uses the original column names for input feature groups from CSV datasets.
            python_function_name (str): Name of Python Function that contains the source code and function arguments.
            python_function_bindings (List): List of python function arguments.
            use_gpu (bool): Whether the feature group needs a gpu or not. Otherwise default to CPU.
            version_limit (int): The number of versions to preserve for the feature group (minimum 30).

        Returns:
            FeatureGroup: The created feature group"""
        return self._call_api('createFeatureGroupFromFunction', 'POST', query_params={}, body={'tableName': table_name, 'functionSourceCode': function_source_code, 'functionName': function_name, 'inputFeatureGroups': input_feature_groups, 'description': description, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements, 'useOriginalCsvNames': use_original_csv_names, 'pythonFunctionName': python_function_name, 'pythonFunctionBindings': python_function_bindings, 'useGpu': use_gpu, 'versionLimit': version_limit}, parse_type=FeatureGroup)

    def create_sampling_feature_group(self, feature_group_id: str, table_name: str, sampling_config: Union[dict, SamplingConfig], description: str = None) -> FeatureGroup:
        """Creates a new Feature Group defined as a sample of rows from another Feature Group.

        For efficiency, sampling is approximate unless otherwise specified. (e.g. the number of rows may vary slightly from what was requested).


        Args:
            feature_group_id (str): The unique ID associated with the pre-existing Feature Group that will be sampled by this new Feature Group. i.e. the input for sampling.
            table_name (str): The unique name to be given to this sampling Feature Group. Can be up to 120 characters long and can only contain alphanumeric characters and underscores.
            sampling_config (SamplingConfig): Dictionary defining the sampling method and its parameters.
            description (str): A human-readable description of this Feature Group.

        Returns:
            FeatureGroup: The created Feature Group."""
        return self._call_api('createSamplingFeatureGroup', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'tableName': table_name, 'samplingConfig': sampling_config, 'description': description}, parse_type=FeatureGroup)

    def create_merge_feature_group(self, source_feature_group_id: str, table_name: str, merge_config: Union[dict, MergeConfig], description: str = None) -> FeatureGroup:
        """Creates a new feature group defined as the union of other feature group versions.

        Args:
            source_feature_group_id (str): Unique string identifier corresponding to the dataset feature group that will have its versions merged into this feature group.
            table_name (str): Unique string identifier to be given to this merge feature group. Can be up to 120 characters long and can only contain alphanumeric characters and underscores.
            merge_config (MergeConfig): JSON object defining the merging method and its parameters.
            description (str): Human-readable description of this feature group.

        Returns:
            FeatureGroup: The created feature group.
Description:
Creates a new feature group defined as the union of other feature group versions."""
        return self._call_api('createMergeFeatureGroup', 'POST', query_params={}, body={'sourceFeatureGroupId': source_feature_group_id, 'tableName': table_name, 'mergeConfig': merge_config, 'description': description}, parse_type=FeatureGroup)

    def create_operator_feature_group(self, source_feature_group_id: str, table_name: str, operator_config: Union[dict, OperatorConfig], description: str = None) -> FeatureGroup:
        """Creates a new Feature Group defined by a pre-defined operator applied to another Feature Group.

        Args:
            source_feature_group_id (str): Unique string identifier corresponding to the Feature Group to which the operator will be applied.
            table_name (str): Unique string identifier for the operator Feature Group. Can be up to 120 characters long and can only contain alphanumeric characters and underscores.
            operator_config (OperatorConfig): The operator config is used to define the operator and its parameters.
            description (str): Human-readable description of the Feature Group.

        Returns:
            FeatureGroup: The created Feature Group."""
        return self._call_api('createOperatorFeatureGroup', 'POST', query_params={}, body={'sourceFeatureGroupId': source_feature_group_id, 'tableName': table_name, 'operatorConfig': operator_config, 'description': description}, parse_type=FeatureGroup)

    def create_snapshot_feature_group(self, feature_group_version: str, table_name: str) -> FeatureGroup:
        """Creates a Snapshot Feature Group corresponding to a specific Feature Group version.

        Args:
            feature_group_version (str): Unique string identifier associated with the Feature Group version being snapshotted.
            table_name (str): Name for the newly created Snapshot Feature Group table. Can be up to 120 characters long and can only contain alphanumeric characters and underscores.

        Returns:
            FeatureGroup: Feature Group corresponding to the newly created Snapshot."""
        return self._call_api('createSnapshotFeatureGroup', 'POST', query_params={}, body={'featureGroupVersion': feature_group_version, 'tableName': table_name}, parse_type=FeatureGroup)

    def create_online_feature_group(self, table_name: str, primary_key: str, description: str = None) -> FeatureGroup:
        """Creates an Online Feature Group.

        Args:
            table_name (str): Name for the newly created feature group. Can be up to 120 characters long and can only contain alphanumeric characters and underscores.
            primary_key (str): The primary key for indexing the online feature group.
            description (str): Human-readable description of the Feature Group.

        Returns:
            FeatureGroup: The created online feature group."""
        return self._call_api('createOnlineFeatureGroup', 'POST', query_params={}, body={'tableName': table_name, 'primaryKey': primary_key, 'description': description}, parse_type=FeatureGroup)

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

    def set_feature_group_operator_config(self, feature_group_id: str, operator_config: Union[dict, OperatorConfig]) -> FeatureGroup:
        """Set a OperatorFeatureGroup’s operator config to the values provided.

        Args:
            feature_group_id (str): A unique string identifier associated with the feature group.
            operator_config (OperatorConfig): A dictionary object specifying the pre-defined operations.

        Returns:
            FeatureGroup: The updated FeatureGroup."""
        return self._call_api('setFeatureGroupOperatorConfig', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'operatorConfig': operator_config}, parse_type=FeatureGroup)

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
            table_name (str): The table name of the feature group to nest. Can be up to 120 characters long and can only contain alphanumeric characters and underscores.
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
            table_name (str): The name of the table. Can be up to 120 characters long and can only contain alphanumeric characters and underscores.
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

    def set_feature_type(self, feature_group_id: str, feature: str, feature_type: str, project_id: str = None) -> Schema:
        """Set the type of a feature in a feature group. Specify the feature group ID, feature name, and feature type, and the method will return the new column with the changes reflected.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            feature (str): The name of the feature.
            feature_type (str): The machine learning type of the data in the feature.
            project_id (str): Optional unique ID associated with the project.

        Returns:
            Schema: The feature group after the data_type is applied."""
        return self._call_api('setFeatureType', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'feature': feature, 'featureType': feature_type, 'projectId': project_id}, parse_type=Schema)

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
        """Sets various attributes of the feature group used for primary key, deployment lookups and streaming updates.

        Args:
            feature_group_id (str): Unique string identifier for the feature group.
            primary_key (str): Name of the feature which defines the primary key of the feature group.
            update_timestamp_key (str): Name of the feature which defines the update timestamp of the feature group. Used in concatenation and primary key deduplication.
            lookup_keys (list): List of feature names which can be used in the lookup API to restrict the computation to a set of dataset rows. These feature names have to correspond to underlying dataset columns."""
        return self._call_api('setFeatureGroupIndexingConfig', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'primaryKey': primary_key, 'updateTimestampKey': update_timestamp_key, 'lookupKeys': lookup_keys})

    def execute_async_feature_group_operation(self, query: str = None, fix_query_on_error: bool = False, use_latest_version: bool = True) -> ExecuteFeatureGroupOperation:
        """Starts the execution of fg operation

        Args:
            query (str): The SQL to be executed.
            fix_query_on_error (bool): If enabled, SQL query is auto fixed if parsing fails.
            use_latest_version (bool): If enabled, executes the query on the latest version of the feature group, and if version doesn't exist, FailedDependencyError is sent. If disabled, query is executed considering the latest feature group state irrespective of the latest version of the feature group.

        Returns:
            ExecuteFeatureGroupOperation: A dict that contains the execution status"""
        return self._call_api('executeAsyncFeatureGroupOperation', 'POST', query_params={}, body={'query': query, 'fixQueryOnError': fix_query_on_error, 'useLatestVersion': use_latest_version}, parse_type=ExecuteFeatureGroupOperation)

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

    def update_feature_group_python_function_bindings(self, feature_group_id: str, python_function_bindings: List):
        """Updates an existing Feature Group's Python function bindings from a user-provided Python Function. If a list of feature groups are supplied within the Python function bindings, we will provide DataFrames (Pandas in the case of Python) with the materialized feature groups for those input feature groups as arguments to the function.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            python_function_bindings (List): List of python function arguments."""
        return self._call_api('updateFeatureGroupPythonFunctionBindings', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'pythonFunctionBindings': python_function_bindings})

    def update_feature_group_python_function(self, feature_group_id: str, python_function_name: str, python_function_bindings: List = None, cpu_size: Union[CPUSize, str] = None, memory: Union[MemorySize, str] = None, use_gpu: bool = None, use_original_csv_names: bool = None):
        """Updates an existing Feature Group's python function from a user provided Python Function. If a list of feature groups are supplied within the python function

        bindings, we will provide as arguments to the function DataFrame's (pandas in the case of Python) with the materialized
        feature groups for those input feature groups.


        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            python_function_name (str): The name of the python function to be associated with the feature group.
            python_function_bindings (List): List of python function arguments.
            cpu_size (CPUSize): Size of the CPU for the feature group python function.
            memory (MemorySize): Memory (in GB) for the feature group python function.
            use_gpu (bool): Whether the feature group needs a gpu or not. Otherwise default to CPU.
            use_original_csv_names (bool): If enabled, it uses the original column names for input feature groups from CSV datasets."""
        return self._call_api('updateFeatureGroupPythonFunction', 'PATCH', query_params={}, body={'featureGroupId': feature_group_id, 'pythonFunctionName': python_function_name, 'pythonFunctionBindings': python_function_bindings, 'cpuSize': cpu_size, 'memory': memory, 'useGpu': use_gpu, 'useOriginalCsvNames': use_original_csv_names})

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

    def delete_feature_group_version(self, feature_group_version: str):
        """Deletes a Feature Group Version.

        Args:
            feature_group_version (str): String identifier for the feature group version to be removed."""
        return self._call_api('deleteFeatureGroupVersion', 'DELETE', query_params={'featureGroupVersion': feature_group_version})

    def create_feature_group_version(self, feature_group_id: str, variable_bindings: dict = None) -> FeatureGroupVersion:
        """Creates a snapshot for a specified feature group. Triggers materialization of the feature group. The new version of the feature group is created after it has materialized.

        Args:
            feature_group_id (str): Unique string identifier associated with the feature group.
            variable_bindings (dict): Dictionary defining variable bindings that override parent feature group values.

        Returns:
            FeatureGroupVersion: A feature group version."""
        return self._call_api('createFeatureGroupVersion', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'variableBindings': variable_bindings}, parse_type=FeatureGroupVersion)

    def set_feature_group_export_connector_config(self, feature_group_id: str, feature_group_export_config: Union[dict, FeatureGroupExportConfig] = None):
        """Sets FG export config for the given feature group.

        Args:
            feature_group_id (str): The unique ID associated with the pre-existing Feature Group for which export config is to be set.
            feature_group_export_config (FeatureGroupExportConfig): The export config to be set for the given feature group."""
        return self._call_api('setFeatureGroupExportConnectorConfig', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'featureGroupExportConfig': feature_group_export_config})

    def set_export_on_materialization(self, feature_group_id: str, enable: bool):
        """Can be used to enable or disable exporting feature group data to the export connector associated with the feature group.

        Args:
            feature_group_id (str): The unique ID associated with the pre-existing Feature Group for which export config is to be set.
            enable (bool): If true, will enable exporting feature group to the connector. If false, will disable."""
        return self._call_api('setExportOnMaterialization', 'POST', query_params={}, body={'featureGroupId': feature_group_id, 'enable': enable})

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
        return self._proxy_request('uploadPart', 'POST', query_params={}, data={'uploadId': upload_id, 'partNumber': part_number}, files={'partData': part_data}, parse_type=UploadPart, is_sync=True)

    def mark_upload_complete(self, upload_id: str) -> Upload:
        """Marks an upload process as complete.

        Args:
            upload_id (str): A unique string identifier for the upload process.

        Returns:
            Upload: The upload object associated with the process, containing details of the file."""
        return self._call_api('markUploadComplete', 'POST', query_params={}, body={'uploadId': upload_id}, parse_type=Upload)

    def create_dataset_from_file_connector(self, table_name: str, location: str, file_format: str = None, refresh_schedule: str = None, csv_delimiter: str = None, filename_column: str = None, start_prefix: str = None, until_prefix: str = None, sql_query: str = None, location_date_format: str = None, date_format_lookback_days: int = None, incremental: bool = False, is_documentset: bool = False, extract_bounding_boxes: bool = False, document_processing_config: Union[dict, DatasetDocumentProcessingConfig] = None, merge_file_schemas: bool = False, reference_only_documentset: bool = False, parsing_config: Union[dict, ParsingConfig] = None, version_limit: int = 30) -> Dataset:
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
            sql_query (str): The SQL query to use when fetching data from the specified location. Use `__TABLE__` as a placeholder for the table name. For example: "SELECT * FROM __TABLE__ WHERE event_date > '2021-01-01'". If not provided, the entire dataset from the specified location will be imported.
            location_date_format (str): The date format in which the data is partitioned in the cloud storage location. For example, if the data is partitioned as s3://bucket1/dir1/dir2/event_date=YYYY-MM-DD/dir4/filename.parquet, then the `location_date_format` is YYYY-MM-DD. This format needs to be consistent across all files within the specified location.
            date_format_lookback_days (int): The number of days to look back from the current day for import locations that are date partitioned. For example, import date 2021-06-04 with `date_format_lookback_days` = 3 will retrieve data for all the dates in the range [2021-06-02, 2021-06-04].
            incremental (bool): Signifies if the dataset is an incremental dataset.
            is_documentset (bool): Signifies if the dataset is docstore dataset. A docstore dataset contains documents like images, PDFs, audio files etc. or is tabular data with links to such files.
            extract_bounding_boxes (bool): Signifies whether to extract bounding boxes out of the documents. Only valid if is_documentset if True.
            document_processing_config (DatasetDocumentProcessingConfig): The document processing configuration. Only valid if is_documentset is True.
            merge_file_schemas (bool): Signifies if the merge file schema policy is enabled. If is_documentset is True, this is also set to True by default.
            reference_only_documentset (bool): Signifies if the data reference only policy is enabled.
            parsing_config (ParsingConfig): Custom config for dataset parsing.
            version_limit (int): The number of recent versions to preserve for the dataset (minimum 30).

        Returns:
            Dataset: The dataset created."""
        return self._call_api('createDatasetFromFileConnector', 'POST', query_params={}, body={'tableName': table_name, 'location': location, 'fileFormat': file_format, 'refreshSchedule': refresh_schedule, 'csvDelimiter': csv_delimiter, 'filenameColumn': filename_column, 'startPrefix': start_prefix, 'untilPrefix': until_prefix, 'sqlQuery': sql_query, 'locationDateFormat': location_date_format, 'dateFormatLookbackDays': date_format_lookback_days, 'incremental': incremental, 'isDocumentset': is_documentset, 'extractBoundingBoxes': extract_bounding_boxes, 'documentProcessingConfig': document_processing_config, 'mergeFileSchemas': merge_file_schemas, 'referenceOnlyDocumentset': reference_only_documentset, 'parsingConfig': parsing_config, 'versionLimit': version_limit}, parse_type=Dataset)

    def create_dataset_version_from_file_connector(self, dataset_id: str, location: str = None, file_format: str = None, csv_delimiter: str = None, merge_file_schemas: bool = None, parsing_config: Union[dict, ParsingConfig] = None, sql_query: str = None) -> DatasetVersion:
        """Creates a new version of the specified dataset.

        Args:
            dataset_id (str): Unique string identifier associated with the dataset.
            location (str): External URI to import the dataset from. If not specified, the last location will be used.
            file_format (str): File format to be used. If not specified, the service will try to detect the file format.
            csv_delimiter (str): If the file format is CSV, use a specific CSV delimiter.
            merge_file_schemas (bool): Signifies if the merge file schema policy is enabled.
            parsing_config (ParsingConfig): Custom config for dataset parsing.
            sql_query (str): The SQL query to use when fetching data from the specified location. Use `__TABLE__` as a placeholder for the table name. For example: "SELECT * FROM __TABLE__ WHERE event_date > '2021-01-01'". If not provided, the entire dataset from the specified location will be imported.

        Returns:
            DatasetVersion: The new Dataset Version created."""
        return self._call_api('createDatasetVersionFromFileConnector', 'POST', query_params={'datasetId': dataset_id}, body={'location': location, 'fileFormat': file_format, 'csvDelimiter': csv_delimiter, 'mergeFileSchemas': merge_file_schemas, 'parsingConfig': parsing_config, 'sqlQuery': sql_query}, parse_type=DatasetVersion)

    def create_dataset_from_database_connector(self, table_name: str, database_connector_id: str, object_name: str = None, columns: str = None, query_arguments: str = None, refresh_schedule: str = None, sql_query: str = None, incremental: bool = False, attachment_parsing_config: Union[dict, AttachmentParsingConfig] = None, incremental_database_connector_config: Union[dict, IncrementalDatabaseConnectorConfig] = None, document_processing_config: Union[dict, DatasetDocumentProcessingConfig] = None, version_limit: int = 30) -> Dataset:
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
            attachment_parsing_config (AttachmentParsingConfig): The attachment parsing configuration. Only valid when attachments are being imported, either will take fg name and column name, or we will take list of urls to import (e.g. importing attachments via Salesforce).
            incremental_database_connector_config (IncrementalDatabaseConnectorConfig): The config for incremental datasets. Only valid if incremental is True
            document_processing_config (DatasetDocumentProcessingConfig): The document processing configuration. Only valid when documents are being imported (e.g. importing KnowledgeArticleDescriptions via Salesforce).
            version_limit (int): The number of recent versions to preserve for the dataset (minimum 30).

        Returns:
            Dataset: The created dataset."""
        return self._call_api('createDatasetFromDatabaseConnector', 'POST', query_params={}, body={'tableName': table_name, 'databaseConnectorId': database_connector_id, 'objectName': object_name, 'columns': columns, 'queryArguments': query_arguments, 'refreshSchedule': refresh_schedule, 'sqlQuery': sql_query, 'incremental': incremental, 'attachmentParsingConfig': attachment_parsing_config, 'incrementalDatabaseConnectorConfig': incremental_database_connector_config, 'documentProcessingConfig': document_processing_config, 'versionLimit': version_limit}, parse_type=Dataset)

    def create_dataset_from_application_connector(self, table_name: str, application_connector_id: str, dataset_config: Union[dict, ApplicationConnectorDatasetConfig] = None, refresh_schedule: str = None, version_limit: int = 30) -> Dataset:
        """Creates a dataset from an Application Connector.

        Args:
            table_name (str): Organization-unique table name.
            application_connector_id (str): Unique string identifier of the application connector to download data from.
            dataset_config (ApplicationConnectorDatasetConfig): Dataset config for the application connector.
            refresh_schedule (str): Cron time string format that describes a schedule to retrieve the latest version of the imported dataset. The time is specified in UTC.
            version_limit (int): The number of recent versions to preserve for the dataset (minimum 30).

        Returns:
            Dataset: The created dataset."""
        return self._call_api('createDatasetFromApplicationConnector', 'POST', query_params={}, body={'tableName': table_name, 'applicationConnectorId': application_connector_id, 'datasetConfig': dataset_config, 'refreshSchedule': refresh_schedule, 'versionLimit': version_limit}, parse_type=Dataset)

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

    def create_dataset_version_from_application_connector(self, dataset_id: str, dataset_config: Union[dict, ApplicationConnectorDatasetConfig] = None) -> DatasetVersion:
        """Creates a new version of the specified dataset.

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            dataset_config (ApplicationConnectorDatasetConfig): Dataset config for the application connector. If any of the fields are not specified, the last values will be used.

        Returns:
            DatasetVersion: The new Dataset Version created."""
        return self._call_api('createDatasetVersionFromApplicationConnector', 'POST', query_params={'datasetId': dataset_id}, body={'datasetConfig': dataset_config}, parse_type=DatasetVersion)

    def create_dataset_from_upload(self, table_name: str, file_format: str = None, csv_delimiter: str = None, is_documentset: bool = False, extract_bounding_boxes: bool = False, parsing_config: Union[dict, ParsingConfig] = None, merge_file_schemas: bool = False, document_processing_config: Union[dict, DatasetDocumentProcessingConfig] = None, version_limit: int = 30) -> Upload:
        """Creates a dataset and returns an upload ID that can be used to upload a file.

        Args:
            table_name (str): Organization-unique table name for this dataset.
            file_format (str): The file format of the dataset.
            csv_delimiter (str): If the file format is CSV, use a specific CSV delimiter.
            is_documentset (bool): Signifies if the dataset is a docstore dataset. A docstore dataset contains documents like images, PDFs, audio files etc. or is tabular data with links to such files.
            extract_bounding_boxes (bool): Signifies whether to extract bounding boxes out of the documents. Only valid if is_documentset if True.
            parsing_config (ParsingConfig): Custom config for dataset parsing.
            merge_file_schemas (bool): Signifies whether to merge the schemas of all files in the dataset. If is_documentset is True, this is also set to True by default.
            document_processing_config (DatasetDocumentProcessingConfig): The document processing configuration. Only valid if is_documentset is True.
            version_limit (int): The number of recent versions to preserve for the dataset (minimum 30).

        Returns:
            Upload: A reference to be used when uploading file parts."""
        return self._call_api('createDatasetFromUpload', 'POST', query_params={}, body={'tableName': table_name, 'fileFormat': file_format, 'csvDelimiter': csv_delimiter, 'isDocumentset': is_documentset, 'extractBoundingBoxes': extract_bounding_boxes, 'parsingConfig': parsing_config, 'mergeFileSchemas': merge_file_schemas, 'documentProcessingConfig': document_processing_config, 'versionLimit': version_limit}, parse_type=Upload)

    def create_dataset_version_from_upload(self, dataset_id: str, file_format: str = None) -> Upload:
        """Creates a new version of the specified dataset using a local file upload.

        Args:
            dataset_id (str): Unique string identifier associated with the dataset.
            file_format (str): File format to be used. If not specified, the service will attempt to detect the file format.

        Returns:
            Upload: Token to be used when uploading file parts."""
        return self._call_api('createDatasetVersionFromUpload', 'POST', query_params={'datasetId': dataset_id}, body={'fileFormat': file_format}, parse_type=Upload)

    def create_dataset_version_from_document_reprocessing(self, dataset_id: str, document_processing_config: Union[dict, DatasetDocumentProcessingConfig] = None) -> DatasetVersion:
        """Creates a new dataset version for a source docstore dataset with the provided document processing configuration. This does not re-import the data but uses the same data which is imported in the latest dataset version and only performs document processing on it.

        Args:
            dataset_id (str): The unique ID associated with the dataset to use as the source dataset.
            document_processing_config (DatasetDocumentProcessingConfig): The document processing configuration to use for the new dataset version. If not specified, the document processing configuration from the source dataset will be used.

        Returns:
            DatasetVersion: The new dataset version created."""
        return self._call_api('createDatasetVersionFromDocumentReprocessing', 'POST', query_params={'datasetId': dataset_id}, body={'documentProcessingConfig': document_processing_config}, parse_type=DatasetVersion)

    def create_streaming_dataset(self, table_name: str, primary_key: str = None, update_timestamp_key: str = None, lookup_keys: list = None, version_limit: int = 30) -> Dataset:
        """Creates a streaming dataset. Use a streaming dataset if your dataset is receiving information from multiple sources over an extended period of time.

        Args:
            table_name (str): The feature group table name to create for this dataset.
            primary_key (str): The optional primary key column name for the dataset.
            update_timestamp_key (str): Name of the feature which defines the update timestamp of the feature group. Used in concatenation and primary key deduplication. Only relevant if lookup keys are set.
            lookup_keys (list): List of feature names which can be used in the lookup API to restrict the computation to a set of dataset rows. These feature names have to correspond to underlying dataset columns.
            version_limit (int): The number of recent versions to preserve for the dataset (minimum 30).

        Returns:
            Dataset: The streaming dataset created."""
        return self._call_api('createStreamingDataset', 'POST', query_params={}, body={'tableName': table_name, 'primaryKey': primary_key, 'updateTimestampKey': update_timestamp_key, 'lookupKeys': lookup_keys, 'versionLimit': version_limit}, parse_type=Dataset)

    def create_realtime_content_store(self, table_name: str, application_connector_id: str, dataset_config: Union[dict, ApplicationConnectorDatasetConfig] = None) -> Dataset:
        """Creates a real-time content store dataset.

        Args:
            table_name (str): Organization-unique table name.
            application_connector_id (str): Unique string identifier of the application connector to download data from.
            dataset_config (ApplicationConnectorDatasetConfig): Dataset config for the application connector.

        Returns:
            Dataset: The created dataset."""
        return self._call_api('createRealtimeContentStore', 'POST', query_params={}, body={'tableName': table_name, 'applicationConnectorId': application_connector_id, 'datasetConfig': dataset_config}, parse_type=Dataset)

    def snapshot_streaming_data(self, dataset_id: str) -> DatasetVersion:
        """Snapshots the current data in the streaming dataset.

        Args:
            dataset_id (str): The unique ID associated with the dataset.

        Returns:
            DatasetVersion: The new Dataset Version created by taking a snapshot of the current data in the streaming dataset."""
        return self._call_api('snapshotStreamingData', 'POST', query_params={'datasetId': dataset_id}, body={}, parse_type=DatasetVersion)

    def set_dataset_column_data_type(self, dataset_id: str, column: str, data_type: Union[DataType, str]) -> Dataset:
        """Set a Dataset's column type.

        Args:
            dataset_id (str): The unique ID associated with the dataset.
            column (str): The name of the column.
            data_type (DataType): The type of the data in the column. Note: Some ColumnMappings may restrict the options or explicitly set the DataType.

        Returns:
            Dataset: The dataset and schema after the data type has been set."""
        return self._call_api('setDatasetColumnDataType', 'POST', query_params={'datasetId': dataset_id}, body={'column': column, 'dataType': data_type}, parse_type=Dataset)

    def create_dataset_from_streaming_connector(self, table_name: str, streaming_connector_id: str, dataset_config: Union[dict, StreamingConnectorDatasetConfig] = None, refresh_schedule: str = None, version_limit: int = 30) -> Dataset:
        """Creates a dataset from a Streaming Connector

        Args:
            table_name (str): Organization-unique table name
            streaming_connector_id (str): Unique String Identifier for the Streaming Connector to import the dataset from
            dataset_config (StreamingConnectorDatasetConfig): Streaming dataset config
            refresh_schedule (str): Cron time string format that describes a schedule to retrieve the latest version of the imported dataset. Time is specified in UTC.
            version_limit (int): The number of recent versions to preserve for the dataset (minimum 30).

        Returns:
            Dataset: The created dataset."""
        return self._call_api('createDatasetFromStreamingConnector', 'POST', query_params={}, body={'tableName': table_name, 'streamingConnectorId': streaming_connector_id, 'datasetConfig': dataset_config, 'refreshSchedule': refresh_schedule, 'versionLimit': version_limit}, parse_type=Dataset)

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
            connection_string (str): The Connection String Abacus.AI should use to authenticate when accessing this bucket.

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

    def delete_dataset_version(self, dataset_version: str):
        """Deletes the specified dataset version from the organization.

        Args:
            dataset_version (str): String identifier of the dataset version to delete."""
        return self._call_api('deleteDatasetVersion', 'DELETE', query_params={'datasetVersion': dataset_version})

    def get_docstore_page_data(self, doc_id: str, page: int, document_processing_config: Union[dict, DocumentProcessingConfig] = None, document_processing_version: str = None) -> PageData:
        """Returns the extracted page data for a document page.

        Args:
            doc_id (str): A unique Docstore string identifier for the document.
            page (int): The page number to retrieve. Page numbers start from 0.
            document_processing_config (DocumentProcessingConfig): The document processing configuration to use for returning the data when the document is processed via EXTRACT_DOCUMENT_DATA Feature Group Operator. If Feature Group Operator is not used, this parameter should be kept as None. If Feature Group Operator is used but this parameter is not provided, the latest available data or the default configuration will be used.
            document_processing_version (str): The document processing version to use for returning the data when the document is processed via EXTRACT_DOCUMENT_DATA Feature Group Operator. If Feature Group Operator is not used, this parameter should be kept as None. If Feature Group Operator is used but this parameter is not provided, the latest version will be used.

        Returns:
            PageData: The extracted page data."""
        return self._proxy_request('getDocstorePageData', 'POST', query_params={}, body={'docId': doc_id, 'page': page, 'documentProcessingConfig': document_processing_config, 'documentProcessingVersion': document_processing_version}, parse_type=PageData, is_sync=True)

    def get_docstore_document_data(self, doc_id: str, document_processing_config: Union[dict, DocumentProcessingConfig] = None, document_processing_version: str = None, return_extracted_page_text: bool = False) -> DocumentData:
        """Returns the extracted data for a document.

        Args:
            doc_id (str): A unique Docstore string identifier for the document.
            document_processing_config (DocumentProcessingConfig): The document processing configuration to use for returning the data when the document is processed via EXTRACT_DOCUMENT_DATA Feature Group Operator. If Feature Group Operator is not used, this parameter should be kept as None. If Feature Group Operator is used but this parameter is not provided, the latest available data or the default configuration will be used.
            document_processing_version (str): The document processing version to use for returning the data when the document is processed via EXTRACT_DOCUMENT_DATA Feature Group Operator. If Feature Group Operator is not used, this parameter should be kept as None. If Feature Group Operator is used but this parameter is not provided, the latest version will be used.
            return_extracted_page_text (bool): Specifies whether to include a list of extracted text for each page in the response. Defaults to false if not provided.

        Returns:
            DocumentData: The extracted document data."""
        return self._proxy_request('getDocstoreDocumentData', 'POST', query_params={}, body={'docId': doc_id, 'documentProcessingConfig': document_processing_config, 'documentProcessingVersion': document_processing_version, 'returnExtractedPageText': return_extracted_page_text}, parse_type=DocumentData, is_sync=True)

    def extract_document_data(self, document: io.TextIOBase = None, doc_id: str = None, document_processing_config: Union[dict, DocumentProcessingConfig] = None, start_page: int = None, end_page: int = None, return_extracted_page_text: bool = False) -> DocumentData:
        """Extracts data from a document using either OCR (for scanned documents/images) or embedded text extraction (for digital documents like .docx). Configure the extraction method through DocumentProcessingConfig

        Args:
            document (io.TextIOBase): The document to extract data from. One of document or doc_id must be provided.
            doc_id (str): A unique Docstore string identifier for the document. One of document or doc_id must be provided.
            document_processing_config (DocumentProcessingConfig): The document processing configuration.
            start_page (int): The starting page to extract data from. Pages are indexed starting from 0. If not provided, the first page will be used.
            end_page (int): The last page to extract data from. Pages are indexed starting from 0. If not provided, the last page will be used.
            return_extracted_page_text (bool): Specifies whether to include a list of extracted text for each page in the response. Defaults to false if not provided.

        Returns:
            DocumentData: The extracted document data."""
        return self._proxy_request('ExtractDocumentData', 'POST', query_params={}, data={'docId': doc_id, 'documentProcessingConfig': json.dumps(document_processing_config.to_dict()) if hasattr(document_processing_config, 'to_dict') else json.dumps(document_processing_config), 'startPage': start_page, 'endPage': end_page, 'returnExtractedPageText': return_extracted_page_text}, files={'document': document}, parse_type=DocumentData)

    def get_training_config_options(self, project_id: str, feature_group_ids: List = None, for_retrain: bool = False, current_training_config: Union[dict, TrainingConfig] = None) -> List[TrainingConfigOptions]:
        """Retrieves the full initial description of the model training configuration options available for the specified project. The configuration options available are determined by the use case associated with the specified project. Refer to the [Use Case Documentation]({USE_CASES_URL}) for more information on use cases and use case-specific configuration options.

        Args:
            project_id (str): The unique ID associated with the project.
            feature_group_ids (List): The feature group IDs to be used for training.
            for_retrain (bool): Whether the training config options are used for retraining.
            current_training_config (TrainingConfig): The current state of the training config, with some options set, which shall be used to get new options after refresh. This is `None` by default initially.

        Returns:
            list[TrainingConfigOptions]: An array of options that can be specified when training a model in this project."""
        return self._call_api('getTrainingConfigOptions', 'POST', query_params={}, body={'projectId': project_id, 'featureGroupIds': feature_group_ids, 'forRetrain': for_retrain, 'currentTrainingConfig': current_training_config}, parse_type=TrainingConfigOptions)

    def create_train_test_data_split_feature_group(self, project_id: str, training_config: Union[dict, TrainingConfig], feature_group_ids: List) -> FeatureGroup:
        """Get the train and test data split without training the model. Only supported for models with custom algorithms.

        Args:
            project_id (str): The unique ID associated with the project.
            training_config (TrainingConfig): The training config used to influence how the split is calculated.
            feature_group_ids (List): List of feature group IDs provided by the user, including the required one for data split and others to influence how to split.

        Returns:
            FeatureGroup: The feature group containing the training data and folds information."""
        return self._call_api('createTrainTestDataSplitFeatureGroup', 'POST', query_params={}, body={'projectId': project_id, 'trainingConfig': training_config, 'featureGroupIds': feature_group_ids}, parse_type=FeatureGroup)

    def train_model(self, project_id: str, name: str = None, training_config: Union[dict, TrainingConfig] = None, feature_group_ids: List = None, refresh_schedule: str = None, custom_algorithms: list = None, custom_algorithms_only: bool = False, custom_algorithm_configs: dict = None, builtin_algorithms: list = None, cpu_size: str = None, memory: int = None, algorithm_training_configs: list = None) -> Model:
        """Create a new model and start its training in the given project.

        Args:
            project_id (str): The unique ID associated with the project.
            name (str): The name of the model. Defaults to "<Project Name> Model".
            training_config (TrainingConfig): The training config used to train this model.
            feature_group_ids (List): List of feature group IDs provided by the user to train the model on.
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically retrain the created model.
            custom_algorithms (list): List of user-defined algorithms to train. If not set, the default enabled custom algorithms will be used.
            custom_algorithms_only (bool): Whether to only run custom algorithms.
            custom_algorithm_configs (dict): Configs for each user-defined algorithm; key is the algorithm name, value is the config serialized to JSON.
            builtin_algorithms (list): List of algorithm names or algorithm IDs of the builtin algorithms provided by Abacus.AI to train. If not set, all applicable builtin algorithms will be used.
            cpu_size (str): Size of the CPU for the user-defined algorithms during training.
            memory (int): Memory (in GB) for the user-defined algorithms during training.
            algorithm_training_configs (list): List of algorithm specifc training configs that will be part of the model training AutoML run.

        Returns:
            Model: The new model which is being trained."""
        return self._call_api('trainModel', 'POST', query_params={}, body={'projectId': project_id, 'name': name, 'trainingConfig': training_config, 'featureGroupIds': feature_group_ids, 'refreshSchedule': refresh_schedule, 'customAlgorithms': custom_algorithms, 'customAlgorithmsOnly': custom_algorithms_only, 'customAlgorithmConfigs': custom_algorithm_configs, 'builtinAlgorithms': builtin_algorithms, 'cpuSize': cpu_size, 'memory': memory, 'algorithmTrainingConfigs': algorithm_training_configs}, parse_type=Model)

    def create_model_from_python(self, project_id: str, function_source_code: str, train_function_name: str, training_input_tables: list, predict_function_name: str = None, predict_many_function_name: str = None, initialize_function_name: str = None, name: str = None, cpu_size: str = None, memory: int = None, training_config: Union[dict, TrainingConfig] = None, exclusive_run: bool = False, package_requirements: list = None, use_gpu: bool = False, is_thread_safe: bool = None) -> Model:
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
            exclusive_run (bool): Decides if this model will be run exclusively or along with other Abacus.AI algorithms
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            use_gpu (bool): Whether this model needs gpu
            is_thread_safe (bool): Whether this model is thread safe

        Returns:
            Model: The new model, which has not been trained."""
        return self._call_api('createModelFromPython', 'POST', query_params={}, body={'projectId': project_id, 'functionSourceCode': function_source_code, 'trainFunctionName': train_function_name, 'trainingInputTables': training_input_tables, 'predictFunctionName': predict_function_name, 'predictManyFunctionName': predict_many_function_name, 'initializeFunctionName': initialize_function_name, 'name': name, 'cpuSize': cpu_size, 'memory': memory, 'trainingConfig': training_config, 'exclusiveRun': exclusive_run, 'packageRequirements': package_requirements, 'useGpu': use_gpu, 'isThreadSafe': is_thread_safe}, parse_type=Model)

    def rename_model(self, model_id: str, name: str):
        """Renames a model

        Args:
            model_id (str): Unique identifier of the model to rename.
            name (str): The new name to assign to the model."""
        return self._call_api('renameModel', 'PATCH', query_params={}, body={'modelId': model_id, 'name': name})

    def update_python_model(self, model_id: str, function_source_code: str = None, train_function_name: str = None, predict_function_name: str = None, predict_many_function_name: str = None, initialize_function_name: str = None, training_input_tables: list = None, cpu_size: str = None, memory: int = None, package_requirements: list = None, use_gpu: bool = None, is_thread_safe: bool = None, training_config: Union[dict, TrainingConfig] = None) -> Model:
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
            is_thread_safe (bool): Whether this model is thread safe
            training_config (TrainingConfig): The training config used to train this model.

        Returns:
            Model: The updated model."""
        return self._call_api('updatePythonModel', 'POST', query_params={}, body={'modelId': model_id, 'functionSourceCode': function_source_code, 'trainFunctionName': train_function_name, 'predictFunctionName': predict_function_name, 'predictManyFunctionName': predict_many_function_name, 'initializeFunctionName': initialize_function_name, 'trainingInputTables': training_input_tables, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements, 'useGpu': use_gpu, 'isThreadSafe': is_thread_safe, 'trainingConfig': training_config}, parse_type=Model)

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

    def set_model_training_config(self, model_id: str, training_config: Union[dict, TrainingConfig], feature_group_ids: List = None) -> Model:
        """Edits the default model training config

        Args:
            model_id (str): A unique string identifier of the model to update.
            training_config (TrainingConfig): The training config used to train this model.
            feature_group_ids (List): The list of feature groups used as input to the model.

        Returns:
            Model: The model object corresponding to the updated training config."""
        return self._call_api('setModelTrainingConfig', 'PATCH', query_params={}, body={'modelId': model_id, 'trainingConfig': training_config, 'featureGroupIds': feature_group_ids}, parse_type=Model)

    def set_model_objective(self, model_version: str, metric: str = None):
        """Sets the best model for all model instances of the model based on the specified metric, and updates the training configuration to use the specified metric for any future model versions.

        If metric is set to None, then just use the default selection


        Args:
            model_version (str): The model version to set as the best model.
            metric (str): The metric to use to determine the best model."""
        return self._call_api('setModelObjective', 'POST', query_params={}, body={'modelVersion': model_version, 'metric': metric})

    def set_model_prediction_params(self, model_id: str, prediction_config: dict) -> Model:
        """Sets the model prediction config for the model

        Args:
            model_id (str): Unique string identifier of the model to update.
            prediction_config (dict): Prediction configuration for the model.

        Returns:
            Model: Model object after the prediction configuration is applied."""
        return self._call_api('setModelPredictionParams', 'PATCH', query_params={}, body={'modelId': model_id, 'predictionConfig': prediction_config}, parse_type=Model)

    def retrain_model(self, model_id: str, deployment_ids: List = None, feature_group_ids: List = None, custom_algorithms: list = None, builtin_algorithms: list = None, custom_algorithm_configs: dict = None, cpu_size: str = None, memory: int = None, training_config: Union[dict, TrainingConfig] = None, algorithm_training_configs: list = None) -> Model:
        """Retrains the specified model, with an option to choose the deployments to which the retraining will be deployed.

        Args:
            model_id (str): Unique string identifier of the model to retrain.
            deployment_ids (List): List of unique string identifiers of deployments to automatically deploy to.
            feature_group_ids (List): List of feature group IDs provided by the user to train the model on.
            custom_algorithms (list): List of user-defined algorithms to train. If not set, will honor the runs from the last time and applicable new custom algorithms.
            builtin_algorithms (list): List of algorithm names or algorithm IDs of Abacus.AI built-in algorithms to train. If not set, will honor the runs from the last time and applicable new built-in algorithms.
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

    def export_model_artifact_as_feature_group(self, model_version: str, table_name: str, artifact_type: Union[EvalArtifactType, str] = None) -> FeatureGroup:
        """Exports metric artifact data for a model as a feature group.

        Args:
            model_version (str): Unique string identifier for the version of the model.
            table_name (str): Name of the feature group table to create.
            artifact_type (EvalArtifactType): eval artifact type to export.

        Returns:
            FeatureGroup: The created feature group."""
        return self._call_api('exportModelArtifactAsFeatureGroup', 'POST', query_params={}, body={'modelVersion': model_version, 'tableName': table_name, 'artifactType': artifact_type}, parse_type=FeatureGroup)

    def set_default_model_algorithm(self, model_id: str, algorithm: str = None, data_cluster_type: str = None):
        """Sets the model's algorithm to default for all new deployments

        Args:
            model_id (str): Unique identifier of the model to set.
            algorithm (str): Algorithm to pin in the model.
            data_cluster_type (str): Data cluster type to set the lead model for."""
        return self._call_api('setDefaultModelAlgorithm', 'POST', query_params={}, body={'modelId': model_id, 'algorithm': algorithm, 'dataClusterType': data_cluster_type})

    def get_custom_train_function_info(self, project_id: str, feature_group_names_for_training: list = None, training_data_parameter_name_override: dict = None, training_config: Union[dict, TrainingConfig] = None, custom_algorithm_config: dict = None) -> CustomTrainFunctionInfo:
        """Returns information about how to call the custom train function.

        Args:
            project_id (str): The unique version ID of the project.
            feature_group_names_for_training (list): A list of feature group table names to be used for training.
            training_data_parameter_name_override (dict): Override from feature group type to parameter name in the train function.
            training_config (TrainingConfig): Training config for the options supported by the Abacus.AI platform.
            custom_algorithm_config (dict): User-defined config that can be serialized by JSON.

        Returns:
            CustomTrainFunctionInfo: Information about how to call the customer-provided train function."""
        return self._call_api('getCustomTrainFunctionInfo', 'POST', query_params={}, body={'projectId': project_id, 'featureGroupNamesForTraining': feature_group_names_for_training, 'trainingDataParameterNameOverride': training_data_parameter_name_override, 'trainingConfig': training_config, 'customAlgorithmConfig': custom_algorithm_config}, parse_type=CustomTrainFunctionInfo)

    def export_custom_model_version(self, model_version: str, output_location: str, algorithm: str = None) -> ModelArtifactsExport:
        """Bundle custom model artifacts to a zip file, and export to the specified location.

        Args:
            model_version (str): A unique string identifier for the model version.
            output_location (str): Location to export the model artifacts results. For example, s3://a-bucket/
            algorithm (str): The algorithm to be exported. Optional if there's only one custom algorithm in the model version.

        Returns:
            ModelArtifactsExport: Object describing the export and its status."""
        return self._call_api('exportCustomModelVersion', 'POST', query_params={}, body={'modelVersion': model_version, 'outputLocation': output_location, 'algorithm': algorithm}, parse_type=ModelArtifactsExport)

    def create_model_monitor(self, project_id: str, prediction_feature_group_id: str, training_feature_group_id: str = None, name: str = None, refresh_schedule: str = None, target_value: str = None, target_value_bias: str = None, target_value_performance: str = None, feature_mappings: dict = None, model_id: str = None, training_feature_mappings: dict = None, feature_group_base_monitor_config: dict = None, feature_group_comparison_monitor_config: dict = None, exclude_interactive_performance_analysis: bool = True, exclude_bias_analysis: bool = None, exclude_performance_analysis: bool = None, exclude_feature_drift_analysis: bool = None, exclude_data_integrity_analysis: bool = None) -> ModelMonitor:
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
            exclude_interactive_performance_analysis (bool): Whether to exclude interactive performance analysis. Defaults to True if not provided.
            exclude_bias_analysis (bool): Whether to exclude bias analysis in the model monitor. For default value bias analysis is included.
            exclude_performance_analysis (bool): Whether to exclude performance analysis in the model monitor. For default value performance analysis is included.
            exclude_feature_drift_analysis (bool): Whether to exclude feature drift analysis in the model monitor. For default value feature drift analysis is included.
            exclude_data_integrity_analysis (bool): Whether to exclude data integrity analysis in the model monitor. For default value data integrity analysis is included.

        Returns:
            ModelMonitor: The new model monitor that was created."""
        return self._call_api('createModelMonitor', 'POST', query_params={}, body={'projectId': project_id, 'predictionFeatureGroupId': prediction_feature_group_id, 'trainingFeatureGroupId': training_feature_group_id, 'name': name, 'refreshSchedule': refresh_schedule, 'targetValue': target_value, 'targetValueBias': target_value_bias, 'targetValuePerformance': target_value_performance, 'featureMappings': feature_mappings, 'modelId': model_id, 'trainingFeatureMappings': training_feature_mappings, 'featureGroupBaseMonitorConfig': feature_group_base_monitor_config, 'featureGroupComparisonMonitorConfig': feature_group_comparison_monitor_config, 'excludeInteractivePerformanceAnalysis': exclude_interactive_performance_analysis, 'excludeBiasAnalysis': exclude_bias_analysis, 'excludePerformanceAnalysis': exclude_performance_analysis, 'excludeFeatureDriftAnalysis': exclude_feature_drift_analysis, 'excludeDataIntegrityAnalysis': exclude_data_integrity_analysis}, parse_type=ModelMonitor)

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

    def create_forecasting_monitor(self, project_id: str, name: str, prediction_feature_group_id: str, training_feature_group_id: str, training_forecast_config: Union[dict, ForecastingMonitorConfig], prediction_forecast_config: Union[dict, ForecastingMonitorConfig], forecast_frequency: str, refresh_schedule: str = None) -> ModelMonitor:
        """Runs a forecasting monitor for the specified project.

        Args:
            project_id (str): Unique string identifier of the project.
            name (str): The name you want your model monitor to have. Defaults to "<Project Name> Model Monitor".
            prediction_feature_group_id (str): Unique string identifier of the prediction data feature group.
            training_feature_group_id (str): Unique string identifier of the training data feature group.
            training_forecast_config (ForecastingMonitorConfig): The configuration for the training data.
            prediction_forecast_config (ForecastingMonitorConfig): The configuration for the prediction data.
            forecast_frequency (str): The frequency of the forecast. Defaults to the frequency of the prediction data.
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically rerun the created forecasting monitor.

        Returns:
            ModelMonitor: The new model monitor that was created."""
        return self._call_api('createForecastingMonitor', 'POST', query_params={}, body={'projectId': project_id, 'name': name, 'predictionFeatureGroupId': prediction_feature_group_id, 'trainingFeatureGroupId': training_feature_group_id, 'trainingForecastConfig': training_forecast_config, 'predictionForecastConfig': prediction_forecast_config, 'forecastFrequency': forecast_frequency, 'refreshSchedule': refresh_schedule}, parse_type=ModelMonitor)

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

    def create_holdout_analysis(self, name: str, model_id: str, feature_group_ids: List, model_version: str = None, algorithm: str = None) -> HoldoutAnalysis:
        """Create a holdout analysis for a model

        Args:
            name (str): Name of the holdout analysis
            model_id (str): ID of the model to create a holdout analysis for
            feature_group_ids (List): List of feature group IDs to use for the holdout analysis
            model_version (str): (optional) Version of the model to use for the holdout analysis
            algorithm (str): (optional) ID of algorithm to use for the holdout analysis

        Returns:
            HoldoutAnalysis: The created holdout analysis"""
        return self._call_api('createHoldoutAnalysis', 'POST', query_params={}, body={'name': name, 'modelId': model_id, 'featureGroupIds': feature_group_ids, 'modelVersion': model_version, 'algorithm': algorithm}, parse_type=HoldoutAnalysis)

    def rerun_holdout_analysis(self, holdout_analysis_id: str, model_version: str = None, algorithm: str = None) -> HoldoutAnalysisVersion:
        """Rerun a holdout analysis. A different model version and algorithm can be specified which should be under the same model.

        Args:
            holdout_analysis_id (str): ID of the holdout analysis to rerun
            model_version (str): (optional) Version of the model to use for the holdout analysis
            algorithm (str): (optional) ID of algorithm to use for the holdout analysis

        Returns:
            HoldoutAnalysisVersion: The created holdout analysis version"""
        return self._call_api('rerunHoldoutAnalysis', 'POST', query_params={}, body={'holdoutAnalysisId': holdout_analysis_id, 'modelVersion': model_version, 'algorithm': algorithm}, parse_type=HoldoutAnalysisVersion)

    def create_monitor_alert(self, project_id: str, alert_name: str, condition_config: Union[dict, AlertConditionConfig], action_config: Union[dict, AlertActionConfig], model_monitor_id: str = None, realtime_monitor_id: str = None) -> MonitorAlert:
        """Create a monitor alert for the given conditions and monitor. We can create monitor alert either for model monitor or real-time monitor.

        Args:
            project_id (str): Unique string identifier for the project.
            alert_name (str): Name of the alert.
            condition_config (AlertConditionConfig): Condition to run the actions for the alert.
            action_config (AlertActionConfig): Configuration for the action of the alert.
            model_monitor_id (str): Unique string identifier for the model monitor created under the project.
            realtime_monitor_id (str): Unique string identifier for the real-time monitor for the deployment created under the project.

        Returns:
            MonitorAlert: Object describing the monitor alert."""
        return self._call_api('createMonitorAlert', 'POST', query_params={}, body={'projectId': project_id, 'alertName': alert_name, 'conditionConfig': condition_config, 'actionConfig': action_config, 'modelMonitorId': model_monitor_id, 'realtimeMonitorId': realtime_monitor_id}, parse_type=MonitorAlert)

    def update_monitor_alert(self, monitor_alert_id: str, alert_name: str = None, condition_config: Union[dict, AlertConditionConfig] = None, action_config: Union[dict, AlertActionConfig] = None) -> MonitorAlert:
        """Update monitor alert

        Args:
            monitor_alert_id (str): Unique identifier of the monitor alert.
            alert_name (str): Name of the alert.
            condition_config (AlertConditionConfig): Condition to run the actions for the alert.
            action_config (AlertActionConfig): Configuration for the action of the alert.

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

    def create_prediction_operator(self, name: str, project_id: str, source_code: str = None, predict_function_name: str = None, initialize_function_name: str = None, feature_group_ids: List = None, cpu_size: str = None, memory: int = None, package_requirements: list = None, use_gpu: bool = False) -> PredictionOperator:
        """Create a new prediction operator.

        Args:
            name (str): Name of the prediction operator.
            project_id (str): The unique ID of the associated project.
            source_code (str): Contents of a valid Python source code file. The source code should contain the function `predictFunctionName`, and the function 'initializeFunctionName' if defined.
            predict_function_name (str): Name of the function found in the source code that will be executed to run predictions.
            initialize_function_name (str): Name of the optional initialize function found in the source code. This function will generate anything used by predictions, based on input feature groups.
            feature_group_ids (List): List of feature groups that are supplied to the initialize function as parameters. Each of the parameters are materialized Dataframes. The order should match the initialize function's parameters.
            cpu_size (str): Size of the CPU for the prediction operator.
            memory (int): Memory (in GB) for the  prediction operator.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            use_gpu (bool): Whether this prediction operator needs gpu.

        Returns:
            PredictionOperator: The created prediction operator object."""
        return self._call_api('createPredictionOperator', 'POST', query_params={}, body={'name': name, 'projectId': project_id, 'sourceCode': source_code, 'predictFunctionName': predict_function_name, 'initializeFunctionName': initialize_function_name, 'featureGroupIds': feature_group_ids, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements, 'useGpu': use_gpu}, parse_type=PredictionOperator)

    def update_prediction_operator(self, prediction_operator_id: str, name: str = None, feature_group_ids: List = None, source_code: str = None, initialize_function_name: str = None, predict_function_name: str = None, cpu_size: str = None, memory: int = None, package_requirements: list = None, use_gpu: bool = None) -> PredictionOperator:
        """Update an existing prediction operator. This does not create a new version.

        Args:
            prediction_operator_id (str): The unique ID of the prediction operator.
            name (str): Name of the prediction operator.
            feature_group_ids (List): List of feature groups that are supplied to the initialize function as parameters. Each of the parameters are materialized Dataframes. The order should match the initialize function's parameters.
            source_code (str): Contents of a valid Python source code file. The source code should contain the function `predictFunctionName`, and the function 'initializeFunctionName' if defined.
            initialize_function_name (str): Name of the optional initialize function found in the source code. This function will generate anything used by predictions, based on input feature groups.
            predict_function_name (str): Name of the function found in the source code that will be executed to run predictions.
            cpu_size (str): Size of the CPU for the prediction operator.
            memory (int): Memory (in GB) for the  prediction operator.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']
            use_gpu (bool): Whether this prediction operator needs gpu.

        Returns:
            PredictionOperator: The updated prediction operator object."""
        return self._call_api('updatePredictionOperator', 'POST', query_params={}, body={'predictionOperatorId': prediction_operator_id, 'name': name, 'featureGroupIds': feature_group_ids, 'sourceCode': source_code, 'initializeFunctionName': initialize_function_name, 'predictFunctionName': predict_function_name, 'cpuSize': cpu_size, 'memory': memory, 'packageRequirements': package_requirements, 'useGpu': use_gpu}, parse_type=PredictionOperator)

    def delete_prediction_operator(self, prediction_operator_id: str):
        """Delete an existing prediction operator.

        Args:
            prediction_operator_id (str): The unique ID of the prediction operator."""
        return self._call_api('deletePredictionOperator', 'DELETE', query_params={'predictionOperatorId': prediction_operator_id})

    def deploy_prediction_operator(self, prediction_operator_id: str, auto_deploy: bool = True) -> Deployment:
        """Deploy the prediction operator.

        Args:
            prediction_operator_id (str): The unique ID of the prediction operator.
            auto_deploy (bool): Flag to enable the automatic deployment when a new prediction operator version is created.

        Returns:
            Deployment: The created deployment object."""
        return self._call_api('deployPredictionOperator', 'POST', query_params={}, body={'predictionOperatorId': prediction_operator_id, 'autoDeploy': auto_deploy}, parse_type=Deployment)

    def create_prediction_operator_version(self, prediction_operator_id: str) -> PredictionOperatorVersion:
        """Create a new version of the prediction operator.

        Args:
            prediction_operator_id (str): The unique ID of the prediction operator.

        Returns:
            PredictionOperatorVersion: The created prediction operator version object."""
        return self._call_api('createPredictionOperatorVersion', 'POST', query_params={}, body={'predictionOperatorId': prediction_operator_id}, parse_type=PredictionOperatorVersion)

    def delete_prediction_operator_version(self, prediction_operator_version: str):
        """Delete a prediction operator version.

        Args:
            prediction_operator_version (str): The unique ID of the prediction operator version."""
        return self._call_api('deletePredictionOperatorVersion', 'DELETE', query_params={'predictionOperatorVersion': prediction_operator_version})

    def create_deployment(self, name: str = None, model_id: str = None, model_version: str = None, algorithm: str = None, feature_group_id: str = None, project_id: str = None, description: str = None, calls_per_second: int = None, auto_deploy: bool = True, start: bool = True, enable_batch_streaming_updates: bool = False, skip_metrics_check: bool = False, model_deployment_config: dict = None) -> Deployment:
        """Creates a deployment with the specified name and description for the specified model or feature group.

        A Deployment makes the trained model or feature group available for prediction requests.


        Args:
            name (str): The name of the deployment.
            model_id (str): The unique ID associated with the model.
            model_version (str): The unique ID associated with the model version to deploy.
            algorithm (str): The unique ID associated with the algorithm to deploy.
            feature_group_id (str): The unique ID associated with a feature group.
            project_id (str): The unique ID associated with a project.
            description (str): The description for the deployment.
            calls_per_second (int): The number of calls per second the deployment can handle.
            auto_deploy (bool): Flag to enable the automatic deployment when a new Model Version finishes training.
            start (bool): If true, will start the deployment; otherwise will create offline
            enable_batch_streaming_updates (bool): Flag to enable marking the feature group deployment to have a background process cache streamed in rows for quicker lookup.
            skip_metrics_check (bool): Flag to skip metric regression with this current deployment
            model_deployment_config (dict): The deployment config for model to deploy

        Returns:
            Deployment: The new model or feature group deployment."""
        return self._call_api('createDeployment', 'POST', query_params={}, body={'name': name, 'modelId': model_id, 'modelVersion': model_version, 'algorithm': algorithm, 'featureGroupId': feature_group_id, 'projectId': project_id, 'description': description, 'callsPerSecond': calls_per_second, 'autoDeploy': auto_deploy, 'start': start, 'enableBatchStreamingUpdates': enable_batch_streaming_updates, 'skipMetricsCheck': skip_metrics_check, 'modelDeploymentConfig': model_deployment_config}, parse_type=Deployment)

    def create_deployment_token(self, project_id: str, name: str = None) -> DeploymentAuthToken:
        """Creates a deployment token for the specified project.

        Deployment tokens are used to authenticate requests to the prediction APIs and are scoped to the project level.


        Args:
            project_id (str): The unique string identifier associated with the project.
            name (str): The name of the deployment token.

        Returns:
            DeploymentAuthToken: The deployment token."""
        return self._call_api('createDeploymentToken', 'POST', query_params={}, body={'projectId': project_id, 'name': name}, parse_type=DeploymentAuthToken)

    def update_deployment(self, deployment_id: str, description: str = None, auto_deploy: bool = None, skip_metrics_check: bool = None):
        """Updates a deployment's properties.

        Args:
            deployment_id (str): Unique identifier of the deployment to update.
            description (str): The new description for the deployment.
            auto_deploy (bool): Flag to enable the automatic deployment when a new Model Version finishes training.
            skip_metrics_check (bool): Flag to skip metric regression with this current deployment. This field is only relevant when auto_deploy is on"""
        return self._call_api('updateDeployment', 'PATCH', query_params={'deploymentId': deployment_id}, body={'description': description, 'autoDeploy': auto_deploy, 'skipMetricsCheck': skip_metrics_check})

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

    def set_deployment_prediction_operator_version(self, deployment_id: str, prediction_operator_version: str):
        """Promotes a prediction operator version to be served in the deployment.

        Args:
            deployment_id (str): Unique string identifier for the deployment.
            prediction_operator_version (str): Unique string identifier for the prediction operator version."""
        return self._call_api('setDeploymentPredictionOperatorVersion', 'PATCH', query_params={'deploymentId': deployment_id}, body={'predictionOperatorVersion': prediction_operator_version})

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

    def set_default_prediction_arguments(self, deployment_id: str, prediction_arguments: Union[dict, PredictionArguments], set_as_override: bool = False) -> Deployment:
        """Sets the deployment config.

        Args:
            deployment_id (str): The unique identifier for a deployment created under the project.
            prediction_arguments (PredictionArguments): The prediction arguments to set.
            set_as_override (bool): If True, use these arguments as overrides instead of defaults for predict calls

        Returns:
            Deployment: description of the updated deployment."""
        return self._call_api('setDefaultPredictionArguments', 'POST', query_params={'deploymentId': deployment_id}, body={'predictionArguments': prediction_arguments, 'setAsOverride': set_as_override}, parse_type=Deployment)

    def create_deployment_alert(self, deployment_id: str, alert_name: str, condition_config: Union[dict, AlertConditionConfig], action_config: Union[dict, AlertActionConfig]) -> MonitorAlert:
        """Create a deployment alert for the given conditions.

        Only support batch prediction usage now.


        Args:
            deployment_id (str): Unique string identifier for the deployment.
            alert_name (str): Name of the alert.
            condition_config (AlertConditionConfig): Condition to run the actions for the alert.
            action_config (AlertActionConfig): Configuration for the action of the alert.

        Returns:
            MonitorAlert: Object describing the deployment alert."""
        return self._call_api('createDeploymentAlert', 'POST', query_params={'deploymentId': deployment_id}, body={'alertName': alert_name, 'conditionConfig': condition_config, 'actionConfig': action_config}, parse_type=MonitorAlert)

    def create_realtime_monitor(self, deployment_id: str, realtime_monitor_schedule: str = None, lookback_time: int = None) -> RealtimeMonitor:
        """Real time monitors compute and monitor metrics of real time prediction data.

        Args:
            deployment_id (str): Unique string identifier for the deployment.
            realtime_monitor_schedule (str): The cron expression for triggering monitor.
            lookback_time (int): Lookback time (in seconds) for each monitor trigger

        Returns:
            RealtimeMonitor: Object describing the real-time monitor."""
        return self._call_api('createRealtimeMonitor', 'POST', query_params={'deploymentId': deployment_id}, body={'realtimeMonitorSchedule': realtime_monitor_schedule, 'lookbackTime': lookback_time}, parse_type=RealtimeMonitor)

    def update_realtime_monitor(self, realtime_monitor_id: str, realtime_monitor_schedule: str = None, lookback_time: float = None) -> RealtimeMonitor:
        """Update the real-time monitor associated with the real-time monitor id.

        Args:
            realtime_monitor_id (str): Unique string identifier for the real-time monitor.
            realtime_monitor_schedule (str): The cron expression for triggering monitor
            lookback_time (float): Lookback time (in seconds) for each monitor trigger

        Returns:
            RealtimeMonitor: Object describing the realtime monitor."""
        return self._call_api('updateRealtimeMonitor', 'POST', query_params={}, body={'realtimeMonitorId': realtime_monitor_id, 'realtimeMonitorSchedule': realtime_monitor_schedule, 'lookbackTime': lookback_time}, parse_type=RealtimeMonitor)

    def delete_realtime_monitor(self, realtime_monitor_id: str):
        """Delete the real-time monitor associated with the real-time monitor id.

        Args:
            realtime_monitor_id (str): Unique string identifier for the real-time monitor."""
        return self._call_api('deleteRealtimeMonitor', 'DELETE', query_params={'realtimeMonitorId': realtime_monitor_id})

    def create_refresh_policy(self, name: str, cron: str, refresh_type: str, project_id: str = None, dataset_ids: List = [], feature_group_id: str = None, model_ids: List = [], deployment_ids: List = [], batch_prediction_ids: List = [], model_monitor_ids: List = [], notebook_id: str = None, prediction_operator_id: str = None, feature_group_export_config: Union[dict, FeatureGroupExportConfig] = None) -> RefreshPolicy:
        """Creates a refresh policy with a particular cron pattern and refresh type. The cron is specified in UTC time.

        A refresh policy allows for the scheduling of a set of actions at regular intervals. This can be useful for periodically updating data that needs to be re-imported into the project for retraining.


        Args:
            name (str): The name of the refresh policy.
            cron (str): A cron-like string specifying the frequency of the refresh policy in UTC time.
            refresh_type (str): The refresh type used to determine what is being refreshed, such as a single dataset, dataset and model, or more.
            project_id (str): Optionally, a project ID can be specified so that all datasets, models, deployments, batch predictions, prediction metrics, model monitrs, and notebooks are captured at the instant the policy was created.
            dataset_ids (List): Comma-separated list of dataset IDs.
            feature_group_id (str): Feature Group ID associated with refresh policy.
            model_ids (List): Comma-separated list of model IDs.
            deployment_ids (List): Comma-separated list of deployment IDs.
            batch_prediction_ids (List): Comma-separated list of batch prediction IDs.
            model_monitor_ids (List): Comma-separated list of model monitor IDs.
            notebook_id (str): Notebook ID associated with refresh policy.
            prediction_operator_id (str): Prediction Operator ID associated with refresh policy.
            feature_group_export_config (FeatureGroupExportConfig): Feature group export configuration.

        Returns:
            RefreshPolicy: The created refresh policy."""
        return self._call_api('createRefreshPolicy', 'POST', query_params={}, body={'name': name, 'cron': cron, 'refreshType': refresh_type, 'projectId': project_id, 'datasetIds': dataset_ids, 'featureGroupId': feature_group_id, 'modelIds': model_ids, 'deploymentIds': deployment_ids, 'batchPredictionIds': batch_prediction_ids, 'modelMonitorIds': model_monitor_ids, 'notebookId': notebook_id, 'predictionOperatorId': prediction_operator_id, 'featureGroupExportConfig': feature_group_export_config}, parse_type=RefreshPolicy)

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
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('lookupFeatures', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'limitResults': limit_results, 'resultColumns': result_columns}, server_override=prediction_url)

    def predict(self, deployment_token: str, deployment_id: str, query_data: dict, **kwargs) -> Dict:
        """Returns a prediction for Predictive Modeling

        Args:
            deployment_token (str): A deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, and is safe to embed in an application or website.
            deployment_id (str): A unique identifier for a deployment created under the project.
            query_data (dict): A dictionary where the key is the column name (e.g. a column with name 'user_id' in the dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed, and the value is the unique value of the same entity."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predict', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id, **kwargs}, body={'queryData': query_data, **kwargs}, server_override=prediction_url)

    def predict_multiple(self, deployment_token: str, deployment_id: str, query_data: list) -> Dict:
        """Returns a list of predictions for predictive modeling.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, and is safe to embed in an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            query_data (list): A list of dictionaries, where the 'key' is the column name (e.g. a column with name 'user_id' in the dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed, and the 'value' is the unique value of the same entity."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predictMultiple', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def predict_from_datasets(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns a list of predictions for Predictive Modeling.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            query_data (dict): A dictionary where the 'key' is the source dataset name, and the 'value' is a list of records corresponding to the dataset rows."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
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
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predictLead', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'explainPredictions': explain_predictions, 'explainerType': explainer_type}, server_override=prediction_url)

    def predict_churn(self, deployment_token: str, deployment_id: str, query_data: dict, explain_predictions: bool = False, explainer_type: str = None) -> Dict:
        """Returns the probability of a user to churn out in response to their interactions with the item/product/service. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'churn_result' mapped to mapping 'CHURNED_YN' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where the 'key' will be the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and the 'value' will be the unique value of the same entity.
            explain_predictions (bool): Will explain predictions for churn
            explainer_type (str): Type of explainer to use for explanations"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predictChurn', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'explainPredictions': explain_predictions, 'explainerType': explainer_type}, server_override=prediction_url)

    def predict_takeover(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns a probability for each class label associated with the types of fraud or a 'yes' or 'no' type label for the possibility of fraud. Note that the inputs to this method, wherever applicable, will be the column names in the dataset mapped to the column mappings in our system (e.g., column 'account_name' mapped to mapping 'ACCOUNT_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): A dictionary containing account activity characteristics (e.g., login id, login duration, login type, IP address, etc.)."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predictTakeover', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def predict_fraud(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns the probability of a transaction performed under a specific account being fraudulent or not. Note that the inputs to this method, wherever applicable, should be the column names in your dataset mapped to the column mappings in our system (e.g. column 'account_number' mapped to the mapping 'ACCOUNT_ID' in our system).

        Args:
            deployment_token (str): A deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique identifier to a deployment created under the project.
            query_data (dict): A dictionary containing transaction attributes (e.g. credit card type, transaction location, transaction amount, etc.)."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predictFraud', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def predict_class(self, deployment_token: str, deployment_id: str, query_data: dict, threshold: float = None, threshold_class: str = None, thresholds: Dict = None, explain_predictions: bool = False, fixed_features: list = None, nested: str = None, explainer_type: str = None) -> Dict:
        """Returns a classification prediction

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model within an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            query_data (dict): A dictionary where the 'Key' is the column name (e.g. a column with the name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and the 'Value' is the unique value of the same entity.
            threshold (float): A float value that is applied on the popular class label.
            threshold_class (str): The label upon which the threshold is added (binary labels only).
            thresholds (Dict): Maps labels to thresholds (multi-label classification only). Defaults to F1 optimal threshold if computed for the given class, else uses 0.5.
            explain_predictions (bool): If True, returns the SHAP explanations for all input features.
            fixed_features (list): A set of input features to treat as constant for explanations - only honored when the explainer type is KERNEL_EXPLAINER
            nested (str): If specified generates prediction delta for each index of the specified nested feature.
            explainer_type (str): The type of explainer to use."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predictClass', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'threshold': threshold, 'thresholdClass': threshold_class, 'thresholds': thresholds, 'explainPredictions': explain_predictions, 'fixedFeatures': fixed_features, 'nested': nested, 'explainerType': explainer_type}, server_override=prediction_url)

    def predict_target(self, deployment_token: str, deployment_id: str, query_data: dict, explain_predictions: bool = False, fixed_features: list = None, nested: str = None, explainer_type: str = None) -> Dict:
        """Returns a prediction from a classification or regression model. Optionally, includes explanations.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            query_data (dict): A dictionary where the 'key' is the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and the 'value' is the unique value of the same entity.
            explain_predictions (bool): If true, returns the SHAP explanations for all input features.
            fixed_features (list): Set of input features to treat as constant for explanations - only honored when the explainer type is KERNEL_EXPLAINER
            nested (str): If specified, generates prediction delta for each index of the specified nested feature.
            explainer_type (str): The type of explainer to use."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predictTarget', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'explainPredictions': explain_predictions, 'fixedFeatures': fixed_features, 'nested': nested, 'explainerType': explainer_type}, server_override=prediction_url)

    def get_anomalies(self, deployment_token: str, deployment_id: str, threshold: float = None, histogram: bool = False) -> io.BytesIO:
        """Returns a list of anomalies from the training dataset.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            threshold (float): The threshold score of what is an anomaly. Valid values are between 0.8 and 0.99.
            histogram (bool): If True, will return a histogram of the distribution of all points."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getAnomalies', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'threshold': threshold, 'histogram': histogram}, server_override=prediction_url)

    def get_timeseries_anomalies(self, deployment_token: str, deployment_id: str, start_timestamp: str = None, end_timestamp: str = None, query_data: dict = None, get_all_item_data: bool = False, series_ids: List = None) -> Dict:
        """Returns a list of anomalous timestamps from the training dataset.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            start_timestamp (str): timestamp from which anomalies have to be detected in the training data
            end_timestamp (str): timestamp to which anomalies have to be detected in the training data
            query_data (dict): additional data on which anomaly detection has to be performed, it can either be a single record or list of records or a json string representing list of records
            get_all_item_data (bool): set this to true if anomaly detection has to be performed on all the data related to input ids
            series_ids (List): list of series ids on which the anomaly detection has to be performed"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getTimeseriesAnomalies', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'startTimestamp': start_timestamp, 'endTimestamp': end_timestamp, 'queryData': query_data, 'getAllItemData': get_all_item_data, 'seriesIds': series_ids}, server_override=prediction_url)

    def is_anomaly(self, deployment_token: str, deployment_id: str, query_data: dict = None) -> Dict:
        """Returns a list of anomaly attributes based on login information for a specified account. Note that the inputs to this method, wherever applicable, should be the column names in the dataset mapped to the column mappings in our system (e.g. column 'account_name' mapped to mapping 'ACCOUNT_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): The input data for the prediction."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('isAnomaly', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def get_event_anomaly_score(self, deployment_token: str, deployment_id: str, query_data: dict = None) -> Dict:
        """Returns an anomaly score for an event.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): The input data for the prediction."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getEventAnomalyScore', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def get_forecast(self, deployment_token: str, deployment_id: str, query_data: dict, future_data: list = None, num_predictions: int = None, prediction_start: str = None, explain_predictions: bool = False, explainer_type: str = None, get_item_data: bool = False) -> Dict:
        """Returns a list of forecasts for a given entity under the specified project deployment. Note that the inputs to the deployed model will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'holiday_yn' mapped to mapping 'FUTURE' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where 'Key' will be the column name (e.g. a column with name 'store_id' in your dataset) mapped to the column mapping ITEM_ID that uniquely identifies the entity against which forecasting is performed and 'Value' will be the unique value of the same entity.
            future_data (list): This will be a list of values known ahead of time that are relevant for forecasting (e.g. State Holidays, National Holidays, etc.). Each element is a dictionary, where the key and the value both will be of type 'str'. For example future data entered for a Store may be [{"Holiday":"No", "Promo":"Yes", "Date": "2015-07-31 00:00:00"}].
            num_predictions (int): The number of timestamps to predict in the future.
            prediction_start (str): The start date for predictions (e.g., "2015-08-01T00:00:00" as input for mid-night of 2015-08-01).
            explain_predictions (bool): Will explain predictions for forecasting
            explainer_type (str): Type of explainer to use for explanations
            get_item_data (bool): Will return the data corresponding to items in query"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getForecast', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'futureData': future_data, 'numPredictions': num_predictions, 'predictionStart': prediction_start, 'explainPredictions': explain_predictions, 'explainerType': explainer_type, 'getItemData': get_item_data}, server_override=prediction_url)

    def get_k_nearest(self, deployment_token: str, deployment_id: str, vector: list, k: int = None, distance: str = None, include_score: bool = False, catalog_id: str = None) -> Dict:
        """Returns the k nearest neighbors for the provided embedding vector.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            vector (list): Input vector to perform the k nearest neighbors with.
            k (int): Overrideable number of items to return.
            distance (str): Specify the distance function to use. Options include “dot“, “cosine“, “euclidean“, and “manhattan“. Default = “dot“
            include_score (bool): If True, will return the score alongside the resulting embedding value.
            catalog_id (str): An optional parameter honored only for embeddings that provide a catalog id"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getKNearest', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'vector': vector, 'k': k, 'distance': distance, 'includeScore': include_score, 'catalogId': catalog_id}, server_override=prediction_url)

    def get_multiple_k_nearest(self, deployment_token: str, deployment_id: str, queries: list):
        """Returns the k nearest neighbors for the queries provided.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            queries (list): List of mappings of format {"catalogId": "cat0", "vectors": [...], "k": 20, "distance": "euclidean"}. See `getKNearest` for additional information about the supported parameters."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getMultipleKNearest', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queries': queries}, server_override=prediction_url)

    def get_labels(self, deployment_token: str, deployment_id: str, query_data: dict, return_extracted_entities: bool = False) -> Dict:
        """Returns a list of scored labels for a document.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): Dictionary where key is "Content" and value is the text from which entities are to be extracted.
            return_extracted_entities (bool): (Optional) If True, will return the extracted entities in simpler format"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getLabels', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'returnExtractedEntities': return_extracted_entities}, server_override=prediction_url)

    def get_entities_from_pdf(self, deployment_token: str, deployment_id: str, pdf: io.TextIOBase = None, doc_id: str = None, return_extracted_features: bool = False, verbose: bool = False, save_extracted_features: bool = None) -> Dict:
        """Extracts text from the provided PDF and returns a list of recognized labels and their scores.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            pdf (io.TextIOBase): (Optional) The pdf to predict on. One of pdf or docId must be specified.
            doc_id (str): (Optional) The pdf to predict on. One of pdf or docId must be specified.
            return_extracted_features (bool): (Optional) If True, will return all extracted features (e.g. all tokens in a page) from the PDF. Default is False.
            verbose (bool): (Optional) If True, will return all the extracted tokens probabilities for all the trained labels. Default is False.
            save_extracted_features (bool): (Optional) If True, will save extracted features (i.e. page tokens) so that they can be fetched using the prediction docId. Default is False."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getEntitiesFromPDF', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={'docId': json.dumps(doc_id) if (doc_id is not None and not isinstance(doc_id, str)) else doc_id, 'returnExtractedFeatures': json.dumps(return_extracted_features) if (return_extracted_features is not None and not isinstance(return_extracted_features, str)) else return_extracted_features, 'verbose': json.dumps(verbose) if (verbose is not None and not isinstance(verbose, str)) else verbose, 'saveExtractedFeatures': json.dumps(save_extracted_features) if (save_extracted_features is not None and not isinstance(save_extracted_features, str)) else save_extracted_features}, files={'pdf': pdf}, server_override=prediction_url)

    def get_recommendations(self, deployment_token: str, deployment_id: str, query_data: dict, num_items: int = None, page: int = None, exclude_item_ids: list = None, score_field: str = None, scaling_factors: list = None, restrict_items: list = None, exclude_items: list = None, explore_fraction: float = None, diversity_attribute_name: str = None, diversity_max_results_per_value: int = None) -> Dict:
        """Returns a list of recommendations for a given user under the specified project deployment. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'time' mapped to mapping 'TIMESTAMP' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where 'Key' will be the column name (e.g. a column with name 'user_name' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the user against which recommendations are made and 'Value' will be the unique value of the same item. For example, if you have the column name 'user_name' mapped to the column mapping 'USER_ID', then the query must have the exact same column name (user_name) as key and the name of the user (John Doe) as value.
            num_items (int): The number of items to recommend on one page. By default, it is set to 50 items per page.
            page (int): The page number to be displayed. For example, let's say that the num_items is set to 10 with the total recommendations list size of 50 recommended items, then an input value of 2 in the 'page' variable will display a list of items that rank from 11th to 20th.
            score_field (str): The relative item scores are returned in a separate field named with the same name as the key (score_field) for this argument.
            scaling_factors (list): It allows you to bias the model towards certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1"], "factor": 1.1}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" in reference to which the model recommendations need to be biased; and the key, "factor" takes the factor by which the item scores are adjusted.  Let's take an example where the input to scaling_factors is [{"column": "VehicleType", "values": ["SUV", "Sedan"], "factor": 1.4}]. After we apply the model to get item probabilities, for every SUV and Sedan in the list, we will multiply the respective probability by 1.1 before sorting. This is particularly useful if there's a type of item that might be less popular but you want to promote it or there's an item that always comes up and you want to demote it.
            restrict_items (list): It allows you to restrict the recommendations to certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1", "value3", ...]}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1", "value3", ...]" to which to restrict the recommendations to. Let's take an example where the input to restrict_items is [{"column": "VehicleType", "values": ["SUV", "Sedan"]}]. This input will restrict the recommendations to SUVs and Sedans. This type of restriction is particularly useful if there's a list of items that you know is of use in some particular scenario and you want to restrict the recommendations only to that list.
            exclude_items (list): It allows you to exclude certain items from the list of recommendations. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1", ...]}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" to exclude from the recommendations. Let's take an example where the input to exclude_items is [{"column": "VehicleType", "values": ["SUV", "Sedan"]}]. The resulting recommendation list will exclude all SUVs and Sedans. This is
            explore_fraction (float): Explore fraction.
            diversity_attribute_name (str): item attribute column name which is used to ensure diversity of prediction results.
            diversity_max_results_per_value (int): maximum number of results per value of diversity_attribute_name."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
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
            deployment_id, deployment_token) if deployment_token else None
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
            deployment_id, deployment_token) if deployment_token else None
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
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getRelatedItems', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'numItems': num_items, 'page': page, 'scalingFactors': scaling_factors, 'restrictItems': restrict_items, 'excludeItems': exclude_items}, server_override=prediction_url)

    def get_chat_response(self, deployment_token: str, deployment_id: str, messages: list, llm_name: str = None, num_completion_tokens: int = None, system_message: str = None, temperature: float = 0.0, filter_key_values: dict = None, search_score_cutoff: float = None, chat_config: dict = None) -> Dict:
        """Return a chat response which continues the conversation based on the input messages and search results.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            messages (list): A list of chronologically ordered messages, starting with a user message and alternating sources. A message is a dict with attributes:     is_user (bool): Whether the message is from the user.      text (str): The message's text.
            llm_name (str): Name of the specific LLM backend to use to power the chat experience
            num_completion_tokens (int): Default for maximum number of tokens for chat answers
            system_message (str): The generative LLM system message
            temperature (float): The generative LLM temperature
            filter_key_values (dict): A dictionary mapping column names to a list of values to restrict the retrieved search results.
            search_score_cutoff (float): Cutoff for the document retriever score. Matching search results below this score will be ignored.
            chat_config (dict): A dictionary specifying the query chat config override."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getChatResponse', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'messages': messages, 'llmName': llm_name, 'numCompletionTokens': num_completion_tokens, 'systemMessage': system_message, 'temperature': temperature, 'filterKeyValues': filter_key_values, 'searchScoreCutoff': search_score_cutoff, 'chatConfig': chat_config}, server_override=prediction_url)

    def get_chat_response_with_binary_data(self, deployment_token: str, deployment_id: str, messages: list, llm_name: str = None, num_completion_tokens: int = None, system_message: str = None, temperature: float = 0.0, filter_key_values: dict = None, search_score_cutoff: float = None, chat_config: dict = None, attachments: None = None) -> Dict:
        """Return a chat response which continues the conversation based on the input messages and search results.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            messages (list): A list of chronologically ordered messages, starting with a user message and alternating sources. A message is a dict with attributes:     is_user (bool): Whether the message is from the user.      text (str): The message's text.
            llm_name (str): Name of the specific LLM backend to use to power the chat experience
            num_completion_tokens (int): Default for maximum number of tokens for chat answers
            system_message (str): The generative LLM system message
            temperature (float): The generative LLM temperature
            filter_key_values (dict): A dictionary mapping column names to a list of values to restrict the retrieved search results.
            search_score_cutoff (float): Cutoff for the document retriever score. Matching search results below this score will be ignored.
            chat_config (dict): A dictionary specifying the query chat config override.
            attachments (None): A dictionary of binary data to use to answer the queries."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getChatResponseWithBinaryData', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={'messages': json.dumps(messages) if (messages is not None and not isinstance(messages, str)) else messages, 'llmName': json.dumps(llm_name) if (llm_name is not None and not isinstance(llm_name, str)) else llm_name, 'numCompletionTokens': json.dumps(num_completion_tokens) if (num_completion_tokens is not None and not isinstance(num_completion_tokens, str)) else num_completion_tokens, 'systemMessage': json.dumps(system_message) if (system_message is not None and not isinstance(system_message, str)) else system_message, 'temperature': json.dumps(temperature) if (temperature is not None and not isinstance(temperature, str)) else temperature, 'filterKeyValues': json.dumps(filter_key_values) if (filter_key_values is not None and not isinstance(filter_key_values, str)) else filter_key_values, 'searchScoreCutoff': json.dumps(search_score_cutoff) if (search_score_cutoff is not None and not isinstance(search_score_cutoff, str)) else search_score_cutoff, 'chatConfig': json.dumps(chat_config) if (chat_config is not None and not isinstance(chat_config, str)) else chat_config}, files=attachments, server_override=prediction_url)

    def get_conversation_response(self, deployment_id: str, message: str, deployment_token: str, deployment_conversation_id: str = None, external_session_id: str = None, llm_name: str = None, num_completion_tokens: int = None, system_message: str = None, temperature: float = 0.0, filter_key_values: dict = None, search_score_cutoff: float = None, chat_config: dict = None, doc_infos: list = None) -> Dict:
        """Return a conversation response which continues the conversation based on the input message and deployment conversation id (if exists).

        Args:
            deployment_id (str): The unique identifier to a deployment created under the project.
            message (str): A message from the user
            deployment_token (str): A token used to authenticate access to deployments created in this project. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_conversation_id (str): The unique identifier of a deployment conversation to continue. If not specified, a new one will be created.
            external_session_id (str): The user supplied unique identifier of a deployment conversation to continue. If specified, we will use this instead of a internal deployment conversation id.
            llm_name (str): Name of the specific LLM backend to use to power the chat experience
            num_completion_tokens (int): Default for maximum number of tokens for chat answers
            system_message (str): The generative LLM system message
            temperature (float): The generative LLM temperature
            filter_key_values (dict): A dictionary mapping column names to a list of values to restrict the retrived search results.
            search_score_cutoff (float): Cutoff for the document retriever score. Matching search results below this score will be ignored.
            chat_config (dict): A dictionary specifiying the query chat config override.
            doc_infos (list): An optional list of documents use for the conversation. A keyword 'doc_id' is expected to be present in each document for retrieving contents from docstore."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getConversationResponse', 'POST', query_params={'deploymentId': deployment_id, 'deploymentToken': deployment_token}, body={'message': message, 'deploymentConversationId': deployment_conversation_id, 'externalSessionId': external_session_id, 'llmName': llm_name, 'numCompletionTokens': num_completion_tokens, 'systemMessage': system_message, 'temperature': temperature, 'filterKeyValues': filter_key_values, 'searchScoreCutoff': search_score_cutoff, 'chatConfig': chat_config, 'docInfos': doc_infos}, server_override=prediction_url)

    def get_conversation_response_with_binary_data(self, deployment_id: str, deployment_token: str, message: str, deployment_conversation_id: str = None, external_session_id: str = None, llm_name: str = None, num_completion_tokens: int = None, system_message: str = None, temperature: float = 0.0, filter_key_values: dict = None, search_score_cutoff: float = None, chat_config: dict = None, attachments: None = None) -> Dict:
        """Return a conversation response which continues the conversation based on the input message and deployment conversation id (if exists).

        Args:
            deployment_id (str): The unique identifier to a deployment created under the project.
            deployment_token (str): A token used to authenticate access to deployments created in this project. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            message (str): A message from the user
            deployment_conversation_id (str): The unique identifier of a deployment conversation to continue. If not specified, a new one will be created.
            external_session_id (str): The user supplied unique identifier of a deployment conversation to continue. If specified, we will use this instead of a internal deployment conversation id.
            llm_name (str): Name of the specific LLM backend to use to power the chat experience
            num_completion_tokens (int): Default for maximum number of tokens for chat answers
            system_message (str): The generative LLM system message
            temperature (float): The generative LLM temperature
            filter_key_values (dict): A dictionary mapping column names to a list of values to restrict the retrived search results.
            search_score_cutoff (float): Cutoff for the document retriever score. Matching search results below this score will be ignored.
            chat_config (dict): A dictionary specifiying the query chat config override.
            attachments (None): A dictionary of binary data to use to answer the queries."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getConversationResponseWithBinaryData', 'POST', query_params={'deploymentId': deployment_id, 'deploymentToken': deployment_token}, data={'message': json.dumps(message) if (message is not None and not isinstance(message, str)) else message, 'deploymentConversationId': json.dumps(deployment_conversation_id) if (deployment_conversation_id is not None and not isinstance(deployment_conversation_id, str)) else deployment_conversation_id, 'externalSessionId': json.dumps(external_session_id) if (external_session_id is not None and not isinstance(external_session_id, str)) else external_session_id, 'llmName': json.dumps(llm_name) if (llm_name is not None and not isinstance(llm_name, str)) else llm_name, 'numCompletionTokens': json.dumps(num_completion_tokens) if (num_completion_tokens is not None and not isinstance(num_completion_tokens, str)) else num_completion_tokens, 'systemMessage': json.dumps(system_message) if (system_message is not None and not isinstance(system_message, str)) else system_message, 'temperature': json.dumps(temperature) if (temperature is not None and not isinstance(temperature, str)) else temperature, 'filterKeyValues': json.dumps(filter_key_values) if (filter_key_values is not None and not isinstance(filter_key_values, str)) else filter_key_values, 'searchScoreCutoff': json.dumps(search_score_cutoff) if (search_score_cutoff is not None and not isinstance(search_score_cutoff, str)) else search_score_cutoff, 'chatConfig': json.dumps(chat_config) if (chat_config is not None and not isinstance(chat_config, str)) else chat_config}, files=attachments, server_override=prediction_url)

    def get_search_results(self, deployment_token: str, deployment_id: str, query_data: dict, num: int = 15) -> Dict:
        """Return the most relevant search results to the search query from the uploaded documents.

        Args:
            deployment_token (str): A token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it can be securely embedded in an application or website.
            deployment_id (str): A unique identifier of a deployment created under the project.
            query_data (dict): A dictionary where the key is "Content" and the value is the text from which entities are to be extracted.
            num (int): Number of search results to return."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getSearchResults', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'num': num}, server_override=prediction_url)

    def get_sentiment(self, deployment_token: str, deployment_id: str, document: str) -> Dict:
        """Predicts sentiment on a document

        Args:
            deployment_token (str): A token used to authenticate access to deployments created in this project. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for a deployment created under this project.
            document (str): The document to be analyzed for sentiment."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getSentiment', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'document': document}, server_override=prediction_url)

    def get_entailment(self, deployment_token: str, deployment_id: str, document: str) -> Dict:
        """Predicts the classification of the document

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            document (str): The document to be classified."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getEntailment', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'document': document}, server_override=prediction_url)

    def get_classification(self, deployment_token: str, deployment_id: str, document: str) -> Dict:
        """Predicts the classification of the document

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            document (str): The document to be classified."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getClassification', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'document': document}, server_override=prediction_url)

    def get_summary(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns a JSON of the predicted summary for the given document. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'text' mapped to mapping 'DOCUMENT' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): Raw data dictionary containing the required document data - must have a key 'document' corresponding to a DOCUMENT type text as value."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getSummary', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def predict_language(self, deployment_token: str, deployment_id: str, query_data: str) -> Dict:
        """Predicts the language of the text

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments within this project, making it safe to embed this model in an application or website.
            deployment_id (str): A unique string identifier for a deployment created under the project.
            query_data (str): The input string to detect."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predictLanguage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def get_assignments(self, deployment_token: str, deployment_id: str, query_data: dict, forced_assignments: dict = None, solve_time_limit_seconds: float = None, include_all_assignments: bool = False) -> Dict:
        """Get all positive assignments that match a query.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it can be safely embedded in an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            query_data (dict): Specifies the set of assignments being requested. The value for the key can be: 1. A simple scalar value, which is matched exactly 2. A list of values, which matches any element in the list 3. A dictionary with keys lower_in/lower_ex and upper_in/upper_ex, which matches values in an inclusive/exclusive range
            forced_assignments (dict): Set of assignments to force and resolve before returning query results.
            solve_time_limit_seconds (float): Maximum time in seconds to spend solving the query.
            include_all_assignments (bool): If True, will return all assignments, including assignments with value 0. Default is False."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getAssignments', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'forcedAssignments': forced_assignments, 'solveTimeLimitSeconds': solve_time_limit_seconds, 'includeAllAssignments': include_all_assignments}, server_override=prediction_url)

    def get_alternative_assignments(self, deployment_token: str, deployment_id: str, query_data: dict, add_constraints: list = None, solve_time_limit_seconds: float = None, best_alternate_only: bool = False) -> Dict:
        """Get alternative positive assignments for given query. Optimal assignments are ignored and the alternative assignments are returned instead.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it can be safely embedded in an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            query_data (dict): Specifies the set of assignments being requested. The value for the key can be: 1. A simple scalar value, which is matched exactly 2. A list of values, which matches any element in the list 3. A dictionary with keys lower_in/lower_ex and upper_in/upper_ex, which matches values in an inclusive/exclusive range
            add_constraints (list): List of constraints dict to apply to the query. The constraint dict should have the following keys: 1. query (dict): Specifies the set of assignment variables involved in the constraint. The format is same as query_data. 2. operator (str): Constraint operator '=' or '<=' or '>='. 3. constant (int): Constraint RHS constant value. 4. coefficient_column (str): Column in Assignment feature group to be used as coefficient for the assignment variables, optional and defaults to 1
            solve_time_limit_seconds (float): Maximum time in seconds to spend solving the query.
            best_alternate_only (bool): When True only the best alternate will be returned, when False multiple alternates are returned"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getAlternativeAssignments', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'addConstraints': add_constraints, 'solveTimeLimitSeconds': solve_time_limit_seconds, 'bestAlternateOnly': best_alternate_only}, server_override=prediction_url)

    def get_optimization_inputs_from_serialized(self, deployment_token: str, deployment_id: str, query_data: dict = None) -> Dict:
        """Get assignments for given query, with new inputs

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it can be safely embedded in an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            query_data (dict): a dictionary with various key: value pairs corresponding to various updated FGs in the FG tree, which we want to update to compute new top level FGs for online solve. (query data will be dict of names: serialized dataframes)"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getOptimizationInputsFromSerialized', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def get_assignments_online_with_new_serialized_inputs(self, deployment_token: str, deployment_id: str, query_data: dict = None, solve_time_limit_seconds: float = None, optimality_gap_limit: float = None) -> Dict:
        """Get assignments for given query, with new inputs

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it can be safely embedded in an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            query_data (dict): a dictionary with assignment, constraint and constraint_equations_df
            solve_time_limit_seconds (float): Maximum time in seconds to spend solving the query.
            optimality_gap_limit (float): Optimality gap we want to come within, after which we accept the solution as valid. (0 means we only want an optimal solution). it is abs(best_solution_found - best_bound) / abs(best_solution_found)"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getAssignmentsOnlineWithNewSerializedInputs', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'solveTimeLimitSeconds': solve_time_limit_seconds, 'optimalityGapLimit': optimality_gap_limit}, server_override=prediction_url)

    def check_constraints(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Check for any constraints violated by the overrides.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model within an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            query_data (dict): Assignment overrides to the solution."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('checkConstraints', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def predict_with_binary_data(self, deployment_token: str, deployment_id: str, blob: io.TextIOBase) -> Dict:
        """Make predictions for a given blob, e.g. image, audio

        Args:
            deployment_token (str): A token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model in an application or website.
            deployment_id (str): A unique identifier to a deployment created under the project.
            blob (io.TextIOBase): The multipart/form-data of the data."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predictWithBinaryData', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={}, files={'blob': blob}, server_override=prediction_url)

    def describe_image(self, deployment_token: str, deployment_id: str, image: io.TextIOBase, categories: list, top_n: int = None) -> Dict:
        """Describe the similarity between an image and a list of categories.

        Args:
            deployment_token (str): Authentication token to access created deployments. This token is only authorized to predict on deployments in the current project, and can be safely embedded in an application or website.
            deployment_id (str): Unique identifier of a deployment created under the project.
            image (io.TextIOBase): Image to describe.
            categories (list): List of candidate categories to compare with the image.
            top_n (int): Return the N most similar categories."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('describeImage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={'categories': json.dumps(categories) if (categories is not None and not isinstance(categories, str)) else categories, 'topN': json.dumps(top_n) if (top_n is not None and not isinstance(top_n, str)) else top_n}, files={'image': image}, server_override=prediction_url)

    def get_text_from_document(self, deployment_token: str, deployment_id: str, document: io.TextIOBase = None, adjust_doc_orientation: bool = False, save_predicted_pdf: bool = False, save_extracted_features: bool = False) -> Dict:
        """Generate text from a document

        Args:
            deployment_token (str): Authentication token to access created deployments. This token is only authorized to predict on deployments in the current project, and can be safely embedded in an application or website.
            deployment_id (str): Unique identifier of a deployment created under the project.
            document (io.TextIOBase): Input document which can be an image, pdf, or word document (Some formats might not be supported yet)
            adjust_doc_orientation (bool): (Optional) whether to detect the document page orientation and rotate it if needed.
            save_predicted_pdf (bool): (Optional) If True, will save the predicted pdf bytes so that they can be fetched using the prediction docId. Default is False.
            save_extracted_features (bool): (Optional) If True, will save extracted features (i.e. page tokens) so that they can be fetched using the prediction docId. Default is False."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getTextFromDocument', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={'adjustDocOrientation': json.dumps(adjust_doc_orientation) if (adjust_doc_orientation is not None and not isinstance(adjust_doc_orientation, str)) else adjust_doc_orientation, 'savePredictedPdf': json.dumps(save_predicted_pdf) if (save_predicted_pdf is not None and not isinstance(save_predicted_pdf, str)) else save_predicted_pdf, 'saveExtractedFeatures': json.dumps(save_extracted_features) if (save_extracted_features is not None and not isinstance(save_extracted_features, str)) else save_extracted_features}, files={'document': document}, server_override=prediction_url)

    def transcribe_audio(self, deployment_token: str, deployment_id: str, audio: io.TextIOBase) -> Dict:
        """Transcribe the audio

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to make predictions on deployments in this project, so it can be safely embedded in an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            audio (io.TextIOBase): The audio to transcribe."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('transcribeAudio', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={}, files={'audio': audio}, server_override=prediction_url)

    def classify_image(self, deployment_token: str, deployment_id: str, image: io.TextIOBase = None, doc_id: str = None) -> Dict:
        """Classify an image.

        Args:
            deployment_token (str): A deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier to a deployment created under the project.
            image (io.TextIOBase): The binary data of the image to classify. One of image or doc_id must be specified.
            doc_id (str): The document ID of the image. One of image or doc_id must be specified."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('classifyImage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={'docId': json.dumps(doc_id) if (doc_id is not None and not isinstance(doc_id, str)) else doc_id}, files={'image': image}, server_override=prediction_url)

    def classify_pdf(self, deployment_token: str, deployment_id: str, pdf: io.TextIOBase = None) -> Dict:
        """Returns a classification prediction from a PDF

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model within an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            pdf (io.TextIOBase): (Optional) The pdf to predict on. One of pdf or docId must be specified."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('classifyPDF', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={}, files={'pdf': pdf}, server_override=prediction_url)

    def get_cluster(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Predicts the cluster for given data.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            query_data (dict): A dictionary where each 'key' represents a column name and its corresponding 'value' represents the value of that column. For Timeseries Clustering, the 'key' should be ITEM_ID, and its value should represent a unique item ID that needs clustering."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getCluster', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def get_objects_from_image(self, deployment_token: str, deployment_id: str, image: io.TextIOBase) -> Dict:
        """Classify an image.

        Args:
            deployment_token (str): A deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier to a deployment created under the project.
            image (io.TextIOBase): The binary data of the image to detect objects from."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getObjectsFromImage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={}, files={'image': image}, server_override=prediction_url)

    def score_image(self, deployment_token: str, deployment_id: str, image: io.TextIOBase) -> Dict:
        """Score on image.

        Args:
            deployment_token (str): A deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier to a deployment created under the project.
            image (io.TextIOBase): The binary data of the image to get the score."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('scoreImage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={}, files={'image': image}, server_override=prediction_url)

    def transfer_style(self, deployment_token: str, deployment_id: str, source_image: io.TextIOBase, style_image: io.TextIOBase) -> io.BytesIO:
        """Change the source image to adopt the visual style from the style image.

        Args:
            deployment_token (str): A token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model in an application or website.
            deployment_id (str): A unique identifier to a deployment created under the project.
            source_image (io.TextIOBase): The source image to apply the makeup.
            style_image (io.TextIOBase): The image that has the style as a reference."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('transferStyle', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={}, files={'sourceImage': source_image, 'styleImage': style_image}, streamable_response=True, server_override=prediction_url)

    def generate_image(self, deployment_token: str, deployment_id: str, query_data: dict) -> io.BytesIO:
        """Generate an image from text prompt.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model within an application or website.
            deployment_id (str): A unique identifier to a deployment created under the project.
            query_data (dict): Specifies the text prompt. For example, {'prompt': 'a cat'}"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('generateImage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, streamable_response=True, server_override=prediction_url)

    def execute_agent(self, deployment_token: str, deployment_id: str, arguments: list = None, keyword_arguments: dict = None) -> Dict:
        """Executes a deployed AI agent function using the arguments as keyword arguments to the agent execute function.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            arguments (list): Positional arguments to the agent execute function.
            keyword_arguments (dict): A dictionary where each 'key' represents the paramter name and its corresponding 'value' represents the value of that parameter for the agent execute function."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('executeAgent', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'arguments': arguments, 'keywordArguments': keyword_arguments}, server_override=prediction_url, timeout=1500)

    def get_matrix_agent_schema(self, deployment_token: str, deployment_id: str, query: str, doc_infos: list = None, deployment_conversation_id: str = None, external_session_id: str = None) -> Dict:
        """Executes a deployed AI agent function using the arguments as keyword arguments to the agent execute function.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            query (str): User input query to initialize the matrix computation.
            doc_infos (list): An optional list of documents use for constructing the matrix. A keyword 'doc_id' is expected to be present in each document for retrieving contents from docstore.
            deployment_conversation_id (str): A unique string identifier for the deployment conversation used for the conversation.
            external_session_id (str): A unique string identifier for the session used for the conversation. If both deployment_conversation_id and external_session_id are not provided, a new session will be created."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getMatrixAgentSchema', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'query': query, 'docInfos': doc_infos, 'deploymentConversationId': deployment_conversation_id, 'externalSessionId': external_session_id}, server_override=prediction_url, timeout=1500)

    def execute_conversation_agent(self, deployment_token: str, deployment_id: str, arguments: list = None, keyword_arguments: dict = None, deployment_conversation_id: str = None, external_session_id: str = None, regenerate: bool = False, doc_infos: list = None, agent_workflow_node_id: str = None) -> Dict:
        """Executes a deployed AI agent function using the arguments as keyword arguments to the agent execute function.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            arguments (list): Positional arguments to the agent execute function.
            keyword_arguments (dict): A dictionary where each 'key' represents the paramter name and its corresponding 'value' represents the value of that parameter for the agent execute function.
            deployment_conversation_id (str): A unique string identifier for the deployment conversation used for the conversation.
            external_session_id (str): A unique string identifier for the session used for the conversation. If both deployment_conversation_id and external_session_id are not provided, a new session will be created.
            regenerate (bool): If True, will regenerate the response from the last query.
            doc_infos (list): An optional list of documents use for the conversation. A keyword 'doc_id' is expected to be present in each document for retrieving contents from docstore.
            agent_workflow_node_id (str): An optional agent workflow node id to trigger agent execution from an intermediate node."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('executeConversationAgent', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'arguments': arguments, 'keywordArguments': keyword_arguments, 'deploymentConversationId': deployment_conversation_id, 'externalSessionId': external_session_id, 'regenerate': regenerate, 'docInfos': doc_infos, 'agentWorkflowNodeId': agent_workflow_node_id}, server_override=prediction_url)

    def lookup_matches(self, deployment_token: str, deployment_id: str, data: str = None, filters: dict = None, num: int = None, result_columns: list = None, max_words: int = None, num_retrieval_margin_words: int = None, max_words_per_chunk: int = None, score_multiplier_column: str = None, min_score: float = None, required_phrases: list = None, filter_clause: str = None, crowding_limits: dict = None, include_text_search: bool = False) -> List[DocumentRetrieverLookupResult]:
        """Lookup document retrievers and return the matching documents from the document retriever deployed with given query.

        Original documents are splitted into chunks and stored in the document retriever. This lookup function will return the relevant chunks
        from the document retriever. The returned chunks could be expanded to include more words from the original documents and merged if they
        are overlapping, and permitted by the settings provided. The returned chunks are sorted by relevance.


        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments within this project, making it safe to embed this model in an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            data (str): The query to search for.
            filters (dict): A dictionary mapping column names to a list of values to restrict the retrieved search results.
            num (int): If provided, will limit the number of results to the value specified.
            result_columns (list): If provided, will limit the column properties present in each result to those specified in this list.
            max_words (int): If provided, will limit the total number of words in the results to the value specified.
            num_retrieval_margin_words (int): If provided, will add this number of words from left and right of the returned chunks.
            max_words_per_chunk (int): If provided, will limit the number of words in each chunk to the value specified. If the value provided is smaller than the actual size of chunk on disk, which is determined during document retriever creation, the actual size of chunk will be used. I.e, chunks looked up from document retrievers will not be split into smaller chunks during lookup due to this setting.
            score_multiplier_column (str): If provided, will use the values in this column to modify the relevance score of the returned chunks. Values in this column must be numeric.
            min_score (float): If provided, will filter out the results with score less than the value specified.
            required_phrases (list): If provided, each result will contain at least one of the phrases in the given list. The matching is whitespace and case insensitive.
            filter_clause (str): If provided, filter the results of the query using this sql where clause.
            crowding_limits (dict): A dictionary mapping metadata columns to the maximum number of results per unique value of the column. This is used to ensure diversity of metadata attribute values in the results. If a particular attribute value has already reached its maximum count, further results with that same attribute value will be excluded from the final result set. An entry in the map can also be a map specifying the limit per attribute value rather than a single limit for all values. This allows a per value limit for attributes. If an attribute value is not present in the map its limit defaults to zero.
            include_text_search (bool): If true, combine the ranking of results from a BM25 text search over the documents with the vector search using reciprocal rank fusion. It leverages both lexical and semantic matching for better overall results. It's particularly valuable in professional, technical, or specialized fields where both precision in terminology and understanding of context are important.

        Returns:
            list[DocumentRetrieverLookupResult]: The relevant documentation results found from the document retriever."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('lookupMatches', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'data': data, 'filters': filters, 'num': num, 'resultColumns': result_columns, 'maxWords': max_words, 'numRetrievalMarginWords': num_retrieval_margin_words, 'maxWordsPerChunk': max_words_per_chunk, 'scoreMultiplierColumn': score_multiplier_column, 'minScore': min_score, 'requiredPhrases': required_phrases, 'filterClause': filter_clause, 'crowdingLimits': crowding_limits, 'includeTextSearch': include_text_search}, parse_type=DocumentRetrieverLookupResult, server_override=prediction_url)

    def get_completion(self, deployment_token: str, deployment_id: str, prompt: str) -> Dict:
        """Returns the finetuned LLM generated completion of the prompt.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            prompt (str): The prompt given to the finetuned LLM to generate the completion."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getCompletion', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'prompt': prompt}, server_override=prediction_url)

    def execute_agent_with_binary_data(self, deployment_token: str, deployment_id: str, arguments: list = None, keyword_arguments: dict = None, deployment_conversation_id: str = None, external_session_id: str = None, blobs: None = None) -> Dict:
        """Executes a deployed AI agent function with binary data as inputs.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            arguments (list): Positional arguments to the agent execute function.
            keyword_arguments (dict): A dictionary where each 'key' represents the parameter name and its corresponding 'value' represents the value of that parameter for the agent execute function.
            deployment_conversation_id (str): A unique string identifier for the deployment conversation used for the conversation.
            external_session_id (str): A unique string identifier for the session used for the conversation. If both deployment_conversation_id and external_session_id are not provided, a new session will be created.
            blobs (None): A dictionary of binary data to use as inputs to the agent execute function.

        Returns:
            AgentDataExecutionResult: The result of the agent execution"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('executeAgentWithBinaryData', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={'arguments': json.dumps(arguments) if (arguments is not None and not isinstance(arguments, str)) else arguments, 'keywordArguments': json.dumps(keyword_arguments) if (keyword_arguments is not None and not isinstance(keyword_arguments, str)) else keyword_arguments, 'deploymentConversationId': json.dumps(deployment_conversation_id) if (deployment_conversation_id is not None and not isinstance(deployment_conversation_id, str)) else deployment_conversation_id, 'externalSessionId': json.dumps(external_session_id) if (external_session_id is not None and not isinstance(external_session_id, str)) else external_session_id}, parse_type=AgentDataExecutionResult, files=blobs, server_override=prediction_url, timeout=1500)

    def start_autonomous_agent(self, deployment_token: str, deployment_id: str, arguments: list = None, keyword_arguments: dict = None, save_conversations: bool = True) -> Dict:
        """Starts a deployed Autonomous agent associated with the given deployment_conversation_id using the arguments and keyword arguments as inputs for execute function of trigger node.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, making it safe to embed this model in an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            arguments (list): Positional arguments to the agent execute function.
            keyword_arguments (dict): A dictionary where each 'key' represents the parameter name and its corresponding 'value' represents the value of that parameter for the agent execute function.
            save_conversations (bool): If true then a new conversation will be created for every run of the workflow associated with the agent."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('startAutonomousAgent', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'arguments': arguments, 'keywordArguments': keyword_arguments, 'saveConversations': save_conversations}, server_override=prediction_url, timeout=1500)

    def pause_autonomous_agent(self, deployment_token: str, deployment_id: str, deployment_conversation_id: str) -> Dict:
        """Pauses a deployed Autonomous agent associated with the given deployment_conversation_id.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, making it safe to embed this model in an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            deployment_conversation_id (str): A unique string identifier for the deployment conversation used for the conversation."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('pauseAutonomousAgent', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'deploymentConversationId': deployment_conversation_id}, server_override=prediction_url, timeout=1500)

    def create_batch_prediction(self, deployment_id: str, table_name: str = None, name: str = None, global_prediction_args: Union[dict, BatchPredictionArgs] = None, batch_prediction_args: Union[dict, BatchPredictionArgs] = None, explanations: bool = False, output_format: str = None, output_location: str = None, database_connector_id: str = None, database_output_config: dict = None, refresh_schedule: str = None, csv_input_prefix: str = None, csv_prediction_prefix: str = None, csv_explanations_prefix: str = None, output_includes_metadata: bool = None, result_input_columns: list = None, input_feature_groups: dict = None) -> BatchPrediction:
        """Creates a batch prediction job description for the given deployment.

        Args:
            deployment_id (str): Unique string identifier for the deployment.
            table_name (str): Name of the feature group table to write the results of the batch prediction. Can only be specified if outputLocation and databaseConnectorId are not specified. If tableName is specified, the outputType will be enforced as CSV.
            name (str): Name of the batch prediction job.
            batch_prediction_args (BatchPredictionArgs): Batch Prediction args specific to problem type.
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
            input_feature_groups (dict): A dict of {'<feature_group_type>': '<feature_group_id>'} which overrides the default input data of that type for the Batch Prediction. Default input data is the training data that was used for training the deployed model.

        Returns:
            BatchPrediction: The batch prediction description."""
        return self._call_api('createBatchPrediction', 'POST', query_params={'deploymentId': deployment_id}, body={'tableName': table_name, 'name': name, 'globalPredictionArgs': global_prediction_args, 'batchPredictionArgs': batch_prediction_args, 'explanations': explanations, 'outputFormat': output_format, 'outputLocation': output_location, 'databaseConnectorId': database_connector_id, 'databaseOutputConfig': database_output_config, 'refreshSchedule': refresh_schedule, 'csvInputPrefix': csv_input_prefix, 'csvPredictionPrefix': csv_prediction_prefix, 'csvExplanationsPrefix': csv_explanations_prefix, 'outputIncludesMetadata': output_includes_metadata, 'resultInputColumns': result_input_columns, 'inputFeatureGroups': input_feature_groups}, parse_type=BatchPrediction)

    def start_batch_prediction(self, batch_prediction_id: str) -> BatchPredictionVersion:
        """Creates a new batch prediction version job for a given batch prediction job description.

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction to create a new version of.

        Returns:
            BatchPredictionVersion: The batch prediction version started by this method call."""
        return self._call_api('startBatchPrediction', 'POST', query_params={}, body={'batchPredictionId': batch_prediction_id}, parse_type=BatchPredictionVersion)

    def update_batch_prediction(self, batch_prediction_id: str, deployment_id: str = None, global_prediction_args: Union[dict, BatchPredictionArgs] = None, batch_prediction_args: Union[dict, BatchPredictionArgs] = None, explanations: bool = None, output_format: str = None, csv_input_prefix: str = None, csv_prediction_prefix: str = None, csv_explanations_prefix: str = None, output_includes_metadata: bool = None, result_input_columns: list = None, name: str = None) -> BatchPrediction:
        """Update a batch prediction job description.

        Args:
            batch_prediction_id (str): Unique identifier of the batch prediction.
            deployment_id (str): Unique identifier of the deployment.
            batch_prediction_args (BatchPredictionArgs): Batch Prediction args specific to problem type.
            output_format (str): If specified, sets the format of the batch prediction output (CSV or JSON).
            csv_input_prefix (str): Prefix to prepend to the input columns, only applies when output format is CSV.
            csv_prediction_prefix (str): Prefix to prepend to the prediction columns, only applies when output format is CSV.
            csv_explanations_prefix (str): Prefix to prepend to the explanation columns, only applies when output format is CSV.
            output_includes_metadata (bool): If True, output will contain columns including prediction start time, batch prediction version, and model version.
            result_input_columns (list): If present, will limit result files or feature groups to only include columns present in this list.
            name (str): If present, will rename the batch prediction.

        Returns:
            BatchPrediction: The batch prediction."""
        return self._call_api('updateBatchPrediction', 'POST', query_params={'deploymentId': deployment_id}, body={'batchPredictionId': batch_prediction_id, 'globalPredictionArgs': global_prediction_args, 'batchPredictionArgs': batch_prediction_args, 'explanations': explanations, 'outputFormat': output_format, 'csvInputPrefix': csv_input_prefix, 'csvPredictionPrefix': csv_prediction_prefix, 'csvExplanationsPrefix': csv_explanations_prefix, 'outputIncludesMetadata': output_includes_metadata, 'resultInputColumns': result_input_columns, 'name': name}, parse_type=BatchPrediction)

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

    def append_data(self, feature_group_id: str, streaming_token: str, data: dict):
        """Appends new data into the feature group for a given lookup key recordId.

        Args:
            feature_group_id (str): Unique string identifier for the streaming feature group to record data to.
            streaming_token (str): The streaming token for authenticating requests.
            data (dict): The data to record as a JSON object."""
        prediction_url = self._get_streaming_endpoint(
            streaming_token, feature_group_id=feature_group_id)
        return self._call_api('appendData', 'POST', query_params={'streamingToken': streaming_token}, body={'featureGroupId': feature_group_id, 'data': data}, server_override=prediction_url)

    def append_multiple_data(self, feature_group_id: str, streaming_token: str, data: list):
        """Appends new data into the feature group for a given lookup key recordId.

        Args:
            feature_group_id (str): Unique string identifier of the streaming feature group to record data to.
            streaming_token (str): Streaming token for authenticating requests.
            data (list): Data to record, as a list of JSON objects."""
        prediction_url = self._get_streaming_endpoint(
            streaming_token, feature_group_id=feature_group_id)
        return self._call_api('appendMultipleData', 'POST', query_params={'streamingToken': streaming_token}, body={'featureGroupId': feature_group_id, 'data': data}, server_override=prediction_url)

    def upsert_data(self, feature_group_id: str, data: dict, streaming_token: str = None, blobs: None = None) -> FeatureGroupRow:
        """Update new data into the feature group for a given lookup key record ID if the record ID is found; otherwise, insert new data into the feature group.

        Args:
            feature_group_id (str): A unique string identifier of the online feature group to record data to.
            data (dict): The data to record, in JSON format.
            streaming_token (str): Optional streaming token for authenticating requests if upserting to streaming FG.
            blobs (None): A dictionary of binary data to populate file fields' in data to upsert to the streaming FG.

        Returns:
            FeatureGroupRow: The feature group row that was upserted."""
        return self._proxy_request('upsertData', 'POST', query_params={}, data={'featureGroupId': feature_group_id, 'data': json.dumps(data.to_dict()) if hasattr(data, 'to_dict') else json.dumps(data), 'streamingToken': streaming_token}, files=blobs, parse_type=FeatureGroupRow, is_sync=True)

    def delete_data(self, feature_group_id: str, primary_key: str):
        """Deletes a row from the feature group given the primary key

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
            primary_key (str): The primary key value for which to delete the feature group row"""
        return self._call_api('deleteData', 'DELETE', query_params={'featureGroupId': feature_group_id, 'primaryKey': primary_key})

    def describe_feature_group_row_process_by_key(self, deployment_id: str, primary_key_value: str) -> FeatureGroupRowProcess:
        """Gets the feature group row process.

        Args:
            deployment_id (str): The deployment id
            primary_key_value (str): The primary key value

        Returns:
            FeatureGroupRowProcess: An object representing the feature group row process"""
        return self._call_api('describeFeatureGroupRowProcessByKey', 'POST', query_params={'deploymentId': deployment_id}, body={'primaryKeyValue': primary_key_value}, parse_type=FeatureGroupRowProcess)

    def list_feature_group_row_processes(self, deployment_id: str, limit: int = None, status: str = None) -> List[FeatureGroupRowProcess]:
        """Gets a list of feature group row processes.

        Args:
            deployment_id (str): The deployment id for the process
            limit (int): The maximum number of processes to return. Defaults to None.
            status (str): The status of the processes to return. Defaults to None.

        Returns:
            list[FeatureGroupRowProcess]: A list of object representing the feature group row process"""
        return self._call_api('listFeatureGroupRowProcesses', 'POST', query_params={'deploymentId': deployment_id}, body={'limit': limit, 'status': status}, parse_type=FeatureGroupRowProcess)

    def get_feature_group_row_process_summary(self, deployment_id: str) -> FeatureGroupRowProcessSummary:
        """Gets a summary of the statuses of the individual feature group processes.

        Args:
            deployment_id (str): The deployment id for the process

        Returns:
            FeatureGroupRowProcessSummary: An object representing the summary of the statuses of the individual feature group processes"""
        return self._call_api('getFeatureGroupRowProcessSummary', 'POST', query_params={'deploymentId': deployment_id}, body={}, parse_type=FeatureGroupRowProcessSummary)

    def reset_feature_group_row_process_by_key(self, deployment_id: str, primary_key_value: str) -> FeatureGroupRowProcess:
        """Resets a feature group row process so that it can be reprocessed

        Args:
            deployment_id (str): The deployment id
            primary_key_value (str): The primary key value

        Returns:
            FeatureGroupRowProcess: An object representing the feature group row process."""
        return self._call_api('resetFeatureGroupRowProcessByKey', 'PATCH', query_params={'deploymentId': deployment_id}, body={'primaryKeyValue': primary_key_value}, parse_type=FeatureGroupRowProcess)

    def get_feature_group_row_process_logs_by_key(self, deployment_id: str, primary_key_value: str) -> FeatureGroupRowProcessLogs:
        """Gets the logs for a feature group row process

        Args:
            deployment_id (str): The deployment id
            primary_key_value (str): The primary key value

        Returns:
            FeatureGroupRowProcessLogs: An object representing the logs for the feature group row process"""
        return self._call_api('getFeatureGroupRowProcessLogsByKey', 'POST', query_params={'deploymentId': deployment_id}, body={'primaryKeyValue': primary_key_value}, parse_type=FeatureGroupRowProcessLogs)

    def create_python_function(self, name: str, source_code: str = None, function_name: str = None, function_variable_mappings: List = None, package_requirements: list = None, function_type: str = 'FEATURE_GROUP', description: str = None, examples: dict = None, user_level_connectors: Dict = None, org_level_connectors: List = None, output_variable_mappings: List = None) -> PythonFunction:
        """Creates a custom Python function that is reusable.

        Args:
            name (str): The name to identify the Python function. Must be a valid Python identifier.
            source_code (str): Contents of a valid Python source code file. The source code should contain the transform feature group functions. A list of allowed imports and system libraries for each language is specified in the user functions documentation section.
            function_name (str): The name of the Python function.
            function_variable_mappings (List): List of Python function arguments.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            function_type (str): Type of Python function to create. Default is FEATURE_GROUP, but can also be PLOTLY_FIG.
            description (str): Description of the Python function. This should include details about the function's purpose, expected inputs and outputs, and any important usage considerations or limitations.
            examples (dict): Dictionary containing example use cases and anti-patterns. Should include 'positive_examples' showing recommended usage and 'negative_examples' showing cases to avoid.])
            user_level_connectors (Dict): Dictionary containing user level connectors.
            org_level_connectors (List): List containing organization level connectors.
            output_variable_mappings (List): List of output variable mappings that defines the elements of the function's return value.

        Returns:
            PythonFunction: The Python function that can be used (e.g. for feature group transform)."""
        return self._call_api('createPythonFunction', 'POST', query_params={}, body={'name': name, 'sourceCode': source_code, 'functionName': function_name, 'functionVariableMappings': function_variable_mappings, 'packageRequirements': package_requirements, 'functionType': function_type, 'description': description, 'examples': examples, 'userLevelConnectors': user_level_connectors, 'orgLevelConnectors': org_level_connectors, 'outputVariableMappings': output_variable_mappings}, parse_type=PythonFunction)

    def update_python_function(self, name: str, source_code: str = None, function_name: str = None, function_variable_mappings: List = None, package_requirements: list = None, description: str = None, examples: dict = None, user_level_connectors: Dict = None, org_level_connectors: List = None, output_variable_mappings: List = None) -> PythonFunction:
        """Update custom python function with user inputs for the given python function.

        Args:
            name (str): The name to identify the Python function. Must be a valid Python identifier.
            source_code (str): Contents of a valid Python source code file. The source code should contain the transform feature group functions. A list of allowed imports and system libraries for each language is specified in the user functions documentation section.
            function_name (str): The name of the Python function within `source_code`.
            function_variable_mappings (List): List of arguments required by `function_name`.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            description (str): Description of the Python function. This should include details about the function's purpose, expected inputs and outputs, and any important usage considerations or limitations.
            examples (dict): Dictionary containing example use cases and anti-patterns. Should include 'positive_examples' showing recommended usage and 'negative_examples' showing cases to avoid.
            user_level_connectors (Dict): Dictionary containing user level connectors.
            org_level_connectors (List): List of organization level connectors.
            output_variable_mappings (List): List of output variable mappings that defines the elements of the function's return value.

        Returns:
            PythonFunction: The Python function object."""
        return self._call_api('updatePythonFunction', 'PATCH', query_params={}, body={'name': name, 'sourceCode': source_code, 'functionName': function_name, 'functionVariableMappings': function_variable_mappings, 'packageRequirements': package_requirements, 'description': description, 'examples': examples, 'userLevelConnectors': user_level_connectors, 'orgLevelConnectors': org_level_connectors, 'outputVariableMappings': output_variable_mappings}, parse_type=PythonFunction)

    def delete_python_function(self, name: str):
        """Removes an existing Python function.

        Args:
            name (str): The name to identify the Python function. Must be a valid Python identifier."""
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

    def update_pipeline(self, pipeline_id: str, project_id: str = None, pipeline_variable_mappings: List = None, cron: str = None, is_prod: bool = None) -> Pipeline:
        """Updates a pipeline for executing multiple steps.

        Args:
            pipeline_id (str): The ID of the pipeline to update.
            project_id (str): A unique string identifier for the pipeline.
            pipeline_variable_mappings (List): List of Python function arguments for the pipeline.
            cron (str): A cron-like string specifying the frequency of the scheduled pipeline runs.
            is_prod (bool): Whether the pipeline is a production pipeline or not.

        Returns:
            Pipeline: An object that describes a Pipeline."""
        return self._call_api('updatePipeline', 'PATCH', query_params={}, body={'pipelineId': pipeline_id, 'projectId': project_id, 'pipelineVariableMappings': pipeline_variable_mappings, 'cron': cron, 'isProd': is_prod}, parse_type=Pipeline)

    def rename_pipeline(self, pipeline_id: str, pipeline_name: str) -> Pipeline:
        """Renames a pipeline.

        Args:
            pipeline_id (str): The ID of the pipeline to rename.
            pipeline_name (str): The new name of the pipeline.

        Returns:
            Pipeline: An object that describes a Pipeline."""
        return self._call_api('renamePipeline', 'PATCH', query_params={}, body={'pipelineId': pipeline_id, 'pipelineName': pipeline_name}, parse_type=Pipeline)

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

    def run_pipeline(self, pipeline_id: str, pipeline_variable_mappings: List = None) -> PipelineVersion:
        """Runs a specified pipeline with the arguments provided.

        Args:
            pipeline_id (str): The ID of the pipeline to run.
            pipeline_variable_mappings (List): List of Python function arguments for the pipeline.

        Returns:
            PipelineVersion: The object describing the pipeline"""
        return self._call_api('runPipeline', 'POST', query_params={}, body={'pipelineId': pipeline_id, 'pipelineVariableMappings': pipeline_variable_mappings}, parse_type=PipelineVersion)

    def reset_pipeline_version(self, pipeline_version: str, steps: list = None, include_downstream_steps: bool = True) -> PipelineVersion:
        """Reruns a pipeline version for the given steps and downstream steps if specified.

        Args:
            pipeline_version (str): The id of the pipeline version.
            steps (list): List of pipeline step names to rerun.
            include_downstream_steps (bool): Whether to rerun downstream steps from the steps you have passed

        Returns:
            PipelineVersion: Object describing the pipeline version"""
        return self._call_api('resetPipelineVersion', 'POST', query_params={}, body={'pipelineVersion': pipeline_version, 'steps': steps, 'includeDownstreamSteps': include_downstream_steps}, parse_type=PipelineVersion)

    def create_pipeline_step(self, pipeline_id: str, step_name: str, function_name: str = None, source_code: str = None, step_input_mappings: List = None, output_variable_mappings: List = None, step_dependencies: list = None, package_requirements: list = None, cpu_size: str = None, memory: int = None, timeout: int = None) -> Pipeline:
        """Creates a step in a given pipeline.

        Args:
            pipeline_id (str): The ID of the pipeline to run.
            step_name (str): The name of the step.
            function_name (str): The name of the Python function.
            source_code (str): Contents of a valid Python source code file. The source code should contain the transform feature group functions. A list of allowed imports and system libraries for each language is specified in the user functions documentation section.
            step_input_mappings (List): List of Python function arguments.
            output_variable_mappings (List): List of Python function outputs.
            step_dependencies (list): List of step names this step depends on.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            cpu_size (str): Size of the CPU for the step function.
            memory (int): Memory (in GB) for the step function.
            timeout (int): Timeout for the step in minutes, default is 300 minutes.

        Returns:
            Pipeline: Object describing the pipeline."""
        return self._call_api('createPipelineStep', 'POST', query_params={}, body={'pipelineId': pipeline_id, 'stepName': step_name, 'functionName': function_name, 'sourceCode': source_code, 'stepInputMappings': step_input_mappings, 'outputVariableMappings': output_variable_mappings, 'stepDependencies': step_dependencies, 'packageRequirements': package_requirements, 'cpuSize': cpu_size, 'memory': memory, 'timeout': timeout}, parse_type=Pipeline)

    def delete_pipeline_step(self, pipeline_step_id: str):
        """Deletes a step from a pipeline.

        Args:
            pipeline_step_id (str): The ID of the pipeline step."""
        return self._call_api('deletePipelineStep', 'DELETE', query_params={'pipelineStepId': pipeline_step_id})

    def update_pipeline_step(self, pipeline_step_id: str, function_name: str = None, source_code: str = None, step_input_mappings: List = None, output_variable_mappings: List = None, step_dependencies: list = None, package_requirements: list = None, cpu_size: str = None, memory: int = None, timeout: int = None) -> PipelineStep:
        """Creates a step in a given pipeline.

        Args:
            pipeline_step_id (str): The ID of the pipeline_step to update.
            function_name (str): The name of the Python function.
            source_code (str): Contents of a valid Python source code file. The source code should contain the transform feature group functions. A list of allowed imports and system libraries for each language is specified in the user functions documentation section.
            step_input_mappings (List): List of Python function arguments.
            output_variable_mappings (List): List of Python function outputs.
            step_dependencies (list): List of step names this step depends on.
            package_requirements (list): List of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            cpu_size (str): Size of the CPU for the step function.
            memory (int): Memory (in GB) for the step function.
            timeout (int): Timeout for the pipeline step, default is 300 minutes.

        Returns:
            PipelineStep: Object describing the pipeline."""
        return self._call_api('updatePipelineStep', 'PATCH', query_params={}, body={'pipelineStepId': pipeline_step_id, 'functionName': function_name, 'sourceCode': source_code, 'stepInputMappings': step_input_mappings, 'outputVariableMappings': output_variable_mappings, 'stepDependencies': step_dependencies, 'packageRequirements': package_requirements, 'cpuSize': cpu_size, 'memory': memory, 'timeout': timeout}, parse_type=PipelineStep)

    def rename_pipeline_step(self, pipeline_step_id: str, step_name: str) -> PipelineStep:
        """Renames a step in a given pipeline.

        Args:
            pipeline_step_id (str): The ID of the pipeline_step to update.
            step_name (str): The name of the step.

        Returns:
            PipelineStep: Object describing the pipeline."""
        return self._call_api('renamePipelineStep', 'PATCH', query_params={}, body={'pipelineStepId': pipeline_step_id, 'stepName': step_name}, parse_type=PipelineStep)

    def unset_pipeline_refresh_schedule(self, pipeline_id: str) -> Pipeline:
        """Deletes the refresh schedule for a given pipeline.

        Args:
            pipeline_id (str): The id of the pipeline.

        Returns:
            Pipeline: Object describing the pipeline."""
        return self._call_api('unsetPipelineRefreshSchedule', 'PATCH', query_params={}, body={'pipelineId': pipeline_id}, parse_type=Pipeline)

    def pause_pipeline_refresh_schedule(self, pipeline_id: str) -> Pipeline:
        """Pauses the refresh schedule for a given pipeline.

        Args:
            pipeline_id (str): The id of the pipeline.

        Returns:
            Pipeline: Object describing the pipeline."""
        return self._call_api('pausePipelineRefreshSchedule', 'POST', query_params={}, body={'pipelineId': pipeline_id}, parse_type=Pipeline)

    def resume_pipeline_refresh_schedule(self, pipeline_id: str) -> Pipeline:
        """Resumes the refresh schedule for a given pipeline.

        Args:
            pipeline_id (str): The id of the pipeline.

        Returns:
            Pipeline: Object describing the pipeline."""
        return self._call_api('resumePipelineRefreshSchedule', 'POST', query_params={}, body={'pipelineId': pipeline_id}, parse_type=Pipeline)

    def skip_pending_pipeline_version_steps(self, pipeline_version: str) -> PipelineVersion:
        """Skips pending steps in a pipeline version.

        Args:
            pipeline_version (str): The id of the pipeline version.

        Returns:
            PipelineVersion: Object describing the pipeline version"""
        return self._call_api('skipPendingPipelineVersionSteps', 'POST', query_params={}, body={'pipelineVersion': pipeline_version}, parse_type=PipelineVersion)

    def create_graph_dashboard(self, project_id: str, name: str, python_function_ids: List = None) -> GraphDashboard:
        """Create a plot dashboard given selected python plots

        Args:
            project_id (str): A unique string identifier for the plot dashboard.
            name (str): The name of the dashboard.
            python_function_ids (List): A list of unique string identifiers for the python functions to be used in the graph dashboard.

        Returns:
            GraphDashboard: An object describing the graph dashboard."""
        return self._call_api('createGraphDashboard', 'POST', query_params={}, body={'projectId': project_id, 'name': name, 'pythonFunctionIds': python_function_ids}, parse_type=GraphDashboard)

    def delete_graph_dashboard(self, graph_dashboard_id: str):
        """Deletes a graph dashboard

        Args:
            graph_dashboard_id (str): Unique string identifier for the graph dashboard to be deleted."""
        return self._call_api('deleteGraphDashboard', 'DELETE', query_params={'graphDashboardId': graph_dashboard_id})

    def update_graph_dashboard(self, graph_dashboard_id: str, name: str = None, python_function_ids: List = None) -> GraphDashboard:
        """Updates a graph dashboard

        Args:
            graph_dashboard_id (str): Unique string identifier for the graph dashboard to update.
            name (str): Name of the dashboard.
            python_function_ids (List): List of unique string identifiers for the Python functions to be used in the graph dashboard.

        Returns:
            GraphDashboard: An object describing the graph dashboard."""
        return self._call_api('updateGraphDashboard', 'POST', query_params={}, body={'graphDashboardId': graph_dashboard_id, 'name': name, 'pythonFunctionIds': python_function_ids}, parse_type=GraphDashboard)

    def add_graph_to_dashboard(self, python_function_id: str, graph_dashboard_id: str, function_variable_mappings: List = None, name: str = None) -> GraphDashboard:
        """Add a python plot function to a dashboard

        Args:
            python_function_id (str): Unique string identifier for the Python function.
            graph_dashboard_id (str): Unique string identifier for the graph dashboard to update.
            function_variable_mappings (List): List of arguments to be supplied to the function as parameters, in the format [{'name': 'function_argument', 'variable_type': 'FEATURE_GROUP', 'value': 'name_of_feature_group'}].
            name (str): Name of the added python plot

        Returns:
            GraphDashboard: An object describing the graph dashboard."""
        return self._call_api('addGraphToDashboard', 'POST', query_params={}, body={'pythonFunctionId': python_function_id, 'graphDashboardId': graph_dashboard_id, 'functionVariableMappings': function_variable_mappings, 'name': name}, parse_type=GraphDashboard)

    def update_graph_to_dashboard(self, graph_reference_id: str, function_variable_mappings: List = None, name: str = None) -> GraphDashboard:
        """Update a python plot function to a dashboard

        Args:
            graph_reference_id (str): A unique string identifier for the graph reference.
            function_variable_mappings (List): A list of arguments to be supplied to the Python function as parameters in the format [{'name': 'function_argument', 'variable_type': 'FEATURE_GROUP', 'value': 'name_of_feature_group'}].
            name (str): The updated name for the graph

        Returns:
            GraphDashboard: An object describing the graph dashboard."""
        return self._call_api('updateGraphToDashboard', 'PATCH', query_params={}, body={'graphReferenceId': graph_reference_id, 'functionVariableMappings': function_variable_mappings, 'name': name}, parse_type=GraphDashboard)

    def delete_graph_from_dashboard(self, graph_reference_id: str):
        """Deletes a python plot function from a dashboard

        Args:
            graph_reference_id (str): Unique String Identifier for the graph"""
        return self._call_api('deleteGraphFromDashboard', 'DELETE', query_params={'graphReferenceId': graph_reference_id})

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

    def list_builtin_algorithms(self, project_id: str, feature_group_ids: List, training_config: Union[dict, TrainingConfig] = None) -> List[Algorithm]:
        """Return list of built-in algorithms based on given input data and training config.

        Args:
            project_id (str): Unique string identifier associated with the project.
            feature_group_ids (List): List of feature group IDs specifying input data.
            training_config (TrainingConfig): The training config to be used for model training.

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

    def create_organization_secret(self, secret_key: str, value: str) -> OrganizationSecret:
        """Creates a secret which can be accessed in functions and notebooks.

        Args:
            secret_key (str): The secret key.
            value (str): The secret value.

        Returns:
            OrganizationSecret: The created secret."""
        return self._call_api('createOrganizationSecret', 'POST', query_params={}, body={'secretKey': secret_key, 'value': value}, parse_type=OrganizationSecret)

    def delete_organization_secret(self, secret_key: str):
        """Deletes a secret.

        Args:
            secret_key (str): The secret key."""
        return self._call_api('deleteOrganizationSecret', 'DELETE', query_params={'secretKey': secret_key})

    def update_organization_secret(self, secret_key: str, value: str) -> OrganizationSecret:
        """Updates a secret.

        Args:
            secret_key (str): The secret key.
            value (str): The secret value.

        Returns:
            OrganizationSecret: The updated secret."""
        return self._call_api('updateOrganizationSecret', 'PATCH', query_params={}, body={'secretKey': secret_key, 'value': value}, parse_type=OrganizationSecret)

    def set_natural_language_explanation(self, short_explanation: str, long_explanation: str, feature_group_id: str = None, feature_group_version: str = None, model_id: str = None):
        """Saves the natural language explanation of an artifact with given ID. The artifact can be - Feature Group or Feature Group Version

        Args:
            short_explanation (str): succinct explanation of the artifact with given ID
            long_explanation (str): verbose explanation of the artifact with given ID
            feature_group_id (str): A unique string identifier associated with the Feature Group.
            feature_group_version (str): A unique string identifier associated with the Feature Group Version.
            model_id (str): A unique string identifier associated with the Model."""
        return self._call_api('setNaturalLanguageExplanation', 'POST', query_params={}, body={'shortExplanation': short_explanation, 'longExplanation': long_explanation, 'featureGroupId': feature_group_id, 'featureGroupVersion': feature_group_version, 'modelId': model_id})

    def create_chat_session(self, project_id: str = None, name: str = None) -> ChatSession:
        """Creates a chat session with Data Science Co-pilot.

        Args:
            project_id (str): The unique project identifier this chat session belongs to
            name (str): The name of the chat session. Defaults to the project name.

        Returns:
            ChatSession: The chat session with Data Science Co-pilot"""
        return self._call_api('createChatSession', 'POST', query_params={}, body={'projectId': project_id, 'name': name}, parse_type=ChatSession)

    def delete_chat_message(self, chat_session_id: str, message_index: int):
        """Deletes a message in a chat session and its associated response.

        Args:
            chat_session_id (str): Unique ID of the chat session.
            message_index (int): The index of the chat message within the UI."""
        return self._call_api('deleteChatMessage', 'DELETE', query_params={'chatSessionId': chat_session_id, 'messageIndex': message_index})

    def export_chat_session(self, chat_session_id: str):
        """Exports a chat session to an HTML file

        Args:
            chat_session_id (str): Unique ID of the chat session."""
        return self._call_api('exportChatSession', 'POST', query_params={}, body={'chatSessionId': chat_session_id})

    def rename_chat_session(self, chat_session_id: str, name: str):
        """Renames a chat session with Data Science Co-pilot.

        Args:
            chat_session_id (str): Unique ID of the chat session.
            name (str): The new name of the chat session."""
        return self._call_api('renameChatSession', 'POST', query_params={}, body={'chatSessionId': chat_session_id, 'name': name})

    def suggest_abacus_apis(self, query: str, verbosity: int = 1, limit: int = 5, include_scores: bool = False) -> List[AbacusApi]:
        """Suggests several Abacus APIs that are most relevant to the supplied natural language query.

        Args:
            query (str): The natural language query to find Abacus APIs for
            verbosity (int): The verbosity level of the suggested Abacus APIs. Ranges from 0 to 2, with 0 being the least verbose and 2 being the most verbose.
            limit (int): The maximum number of APIs to return
            include_scores (bool): Whether to include the relevance scores of the suggested APIs

        Returns:
            list[AbacusApi]: A list of suggested Abacus APIs"""
        return self._call_api('suggestAbacusApis', 'POST', query_params={}, body={'query': query, 'verbosity': verbosity, 'limit': limit, 'includeScores': include_scores}, parse_type=AbacusApi)

    def create_deployment_conversation(self, deployment_id: str = None, name: str = None, external_application_id: str = None) -> DeploymentConversation:
        """Creates a deployment conversation.

        Args:
            deployment_id (str): The deployment this conversation belongs to.
            name (str): The name of the conversation.
            external_application_id (str): The external application id associated with the deployment conversation.

        Returns:
            DeploymentConversation: The deployment conversation."""
        return self._proxy_request('createDeploymentConversation', 'POST', query_params={'deploymentId': deployment_id}, body={'name': name, 'externalApplicationId': external_application_id}, parse_type=DeploymentConversation, is_sync=True)

    def delete_deployment_conversation(self, deployment_conversation_id: str, deployment_id: str = None):
        """Delete a Deployment Conversation.

        Args:
            deployment_conversation_id (str): A unique string identifier associated with the deployment conversation.
            deployment_id (str): The deployment this conversation belongs to. This is required if not logged in."""
        return self._proxy_request('deleteDeploymentConversation', 'DELETE', query_params={'deploymentConversationId': deployment_conversation_id, 'deploymentId': deployment_id}, is_sync=True)

    def clear_deployment_conversation(self, deployment_conversation_id: str = None, external_session_id: str = None, deployment_id: str = None, user_message_indices: list = None):
        """Clear the message history of a Deployment Conversation.

        Args:
            deployment_conversation_id (str): A unique string identifier associated with the deployment conversation.
            external_session_id (str): The external session id associated with the deployment conversation.
            deployment_id (str): The deployment this conversation belongs to. This is required if not logged in.
            user_message_indices (list): Optional list of user message indices to clear. The associated bot response will also be cleared. If not provided, all messages will be cleared."""
        return self._proxy_request('clearDeploymentConversation', 'POST', query_params={'deploymentId': deployment_id}, body={'deploymentConversationId': deployment_conversation_id, 'externalSessionId': external_session_id, 'userMessageIndices': user_message_indices}, is_sync=True)

    def set_deployment_conversation_feedback(self, deployment_conversation_id: str, message_index: int, is_useful: bool = None, is_not_useful: bool = None, feedback: str = None, feedback_type: str = None, deployment_id: str = None):
        """Sets a deployment conversation message as useful or not useful

        Args:
            deployment_conversation_id (str): A unique string identifier associated with the deployment conversation.
            message_index (int): The index of the deployment conversation message
            is_useful (bool): If the message is useful. If true, the message is useful. If false, clear the useful flag.
            is_not_useful (bool): If the message is not useful. If true, the message is not useful. If set to false, clear the useful flag.
            feedback (str): Optional feedback on why the message is useful or not useful
            feedback_type (str): Optional feedback type
            deployment_id (str): The deployment this conversation belongs to. This is required if not logged in."""
        return self._call_api('setDeploymentConversationFeedback', 'POST', query_params={'deploymentId': deployment_id}, body={'deploymentConversationId': deployment_conversation_id, 'messageIndex': message_index, 'isUseful': is_useful, 'isNotUseful': is_not_useful, 'feedback': feedback, 'feedbackType': feedback_type})

    def rename_deployment_conversation(self, deployment_conversation_id: str, name: str, deployment_id: str = None):
        """Rename a Deployment Conversation.

        Args:
            deployment_conversation_id (str): A unique string identifier associated with the deployment conversation.
            name (str): The new name of the conversation.
            deployment_id (str): The deployment this conversation belongs to. This is required if not logged in."""
        return self._proxy_request('renameDeploymentConversation', 'POST', query_params={'deploymentId': deployment_id}, body={'deploymentConversationId': deployment_conversation_id, 'name': name}, is_sync=True)

    def create_app_user_group(self, name: str) -> AppUserGroup:
        """Creates a new App User Group. This User Group is used to have permissions to access the external chatbots.

        Args:
            name (str): The name of the App User Group.

        Returns:
            AppUserGroup: The App User Group."""
        return self._call_api('createAppUserGroup', 'POST', query_params={}, body={'name': name}, parse_type=AppUserGroup)

    def delete_app_user_group(self, user_group_id: str):
        """Deletes an App User Group.

        Args:
            user_group_id (str): The ID of the App User Group."""
        return self._call_api('deleteAppUserGroup', 'DELETE', query_params={'userGroupId': user_group_id})

    def invite_users_to_app_user_group(self, user_group_id: str, emails: List) -> ExternalInvite:
        """Invite users to an App User Group. This method will send the specified email addresses an invitation link to join a specific user group.

        This will allow them to use any chatbots that this user group has access to.


        Args:
            user_group_id (str): The ID of the App User Group to invite the user to.
            emails (List): The email addresses to invite to your user group.

        Returns:
            ExternalInvite: The response of the invitation. This will contain the emails that were successfully invited and the emails that were not."""
        return self._call_api('inviteUsersToAppUserGroup', 'POST', query_params={}, body={'userGroupId': user_group_id, 'emails': emails}, parse_type=ExternalInvite)

    def add_users_to_app_user_group(self, user_group_id: str, user_emails: list):
        """Adds users to a App User Group.

        Args:
            user_group_id (str): The ID of the App User Group.
            user_emails (list): The emails of the users to add to the App User Group."""
        return self._call_api('addUsersToAppUserGroup', 'POST', query_params={}, body={'userGroupId': user_group_id, 'userEmails': user_emails})

    def remove_users_from_app_user_group(self, user_group_id: str, user_emails: list):
        """Removes users from an App User Group.

        Args:
            user_group_id (str): The ID of the App User Group.
            user_emails (list): The emails of the users to remove from the App User Group."""
        return self._call_api('removeUsersFromAppUserGroup', 'POST', query_params={}, body={'userGroupId': user_group_id, 'userEmails': user_emails})

    def add_app_user_group_report_permission(self, user_group_id: str):
        """Give the App User Group the permission to view all reports in the corresponding organization.

        Args:
            user_group_id (str): The ID of the App User Group."""
        return self._call_api('addAppUserGroupReportPermission', 'POST', query_params={}, body={'userGroupId': user_group_id})

    def remove_app_user_group_report_permission(self, user_group_id: str):
        """Remove the App User Group's permission toview all reports in the corresponding organization.

        Args:
            user_group_id (str): The ID of the App User Group."""
        return self._call_api('removeAppUserGroupReportPermission', 'POST', query_params={}, body={'userGroupId': user_group_id})

    def add_app_user_group_to_external_application(self, user_group_id: str, external_application_id: str):
        """Adds a permission for an App User Group to access an External Application.

        Args:
            user_group_id (str): The ID of the App User Group.
            external_application_id (str): The ID of the External Application."""
        return self._call_api('addAppUserGroupToExternalApplication', 'POST', query_params={}, body={'userGroupId': user_group_id, 'externalApplicationId': external_application_id})

    def remove_app_user_group_from_external_application(self, user_group_id: str, external_application_id: str):
        """Removes a permission for an App User Group to access an External Application.

        Args:
            user_group_id (str): The ID of the App User Group.
            external_application_id (str): The ID of the External Application."""
        return self._call_api('removeAppUserGroupFromExternalApplication', 'POST', query_params={}, body={'userGroupId': user_group_id, 'externalApplicationId': external_application_id})

    def create_external_application(self, deployment_id: str, name: str = None, description: str = None, logo: str = None, theme: dict = None) -> ExternalApplication:
        """Creates a new External Application from an existing ChatLLM Deployment.

        Args:
            deployment_id (str): The ID of the deployment to use.
            name (str): The name of the External Application. If not provided, the name of the deployment will be used.
            description (str): The description of the External Application. This will be shown to users when they access the External Application. If not provided, the description of the deployment will be used.
            logo (str): The logo to be displayed.
            theme (dict): The visual theme of the External Application.

        Returns:
            ExternalApplication: The newly created External Application."""
        return self._call_api('createExternalApplication', 'POST', query_params={'deploymentId': deployment_id}, body={'name': name, 'description': description, 'logo': logo, 'theme': theme}, parse_type=ExternalApplication)

    def update_external_application(self, external_application_id: str, name: str = None, description: str = None, theme: dict = None, deployment_id: str = None, deployment_conversation_retention_hours: int = None, reset_retention_policy: bool = False) -> ExternalApplication:
        """Updates an External Application.

        Args:
            external_application_id (str): The ID of the External Application.
            name (str): The name of the External Application.
            description (str): The description of the External Application. This will be shown to users when they access the External Application.
            theme (dict): The visual theme of the External Application.
            deployment_id (str): The ID of the deployment to use.
            deployment_conversation_retention_hours (int): The number of hours to retain the conversations for.
            reset_retention_policy (bool): If true, the retention policy will be removed.

        Returns:
            ExternalApplication: The updated External Application."""
        return self._call_api('updateExternalApplication', 'POST', query_params={'deploymentId': deployment_id}, body={'externalApplicationId': external_application_id, 'name': name, 'description': description, 'theme': theme, 'deploymentConversationRetentionHours': deployment_conversation_retention_hours, 'resetRetentionPolicy': reset_retention_policy}, parse_type=ExternalApplication)

    def delete_external_application(self, external_application_id: str):
        """Deletes an External Application.

        Args:
            external_application_id (str): The ID of the External Application."""
        return self._call_api('deleteExternalApplication', 'DELETE', query_params={'externalApplicationId': external_application_id})

    def create_agent(self, project_id: str, function_source_code: str = None, agent_function_name: str = None, name: str = None, memory: int = None, package_requirements: list = [], description: str = None, enable_binary_input: bool = False, evaluation_feature_group_id: str = None, agent_input_schema: dict = None, agent_output_schema: dict = None, workflow_graph: Union[dict, WorkflowGraph] = None, agent_interface: Union[AgentInterface, str] = AgentInterface.DEFAULT, included_modules: List = None, org_level_connectors: List = None, user_level_connectors: Dict = None, initialize_function_name: str = None, initialize_function_code: str = None) -> Agent:
        """Creates a new AI agent using the given agent workflow graph definition.

        Args:
            project_id (str): The unique ID associated with the project.
            name (str): The name you want your agent to have, defaults to "<Project Name> Agent".
            memory (int): Overrides the default memory allocation (in GB) for the agent.
            package_requirements (list): A list of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            description (str): A description of the agent, including its purpose and instructions.
            evaluation_feature_group_id (str): The ID of the feature group to use for evaluation.
            workflow_graph (WorkflowGraph): The workflow graph for the agent.
            agent_interface (AgentInterface): The interface that the agent will be deployed with.
            included_modules (List): A list of user created custom modules to include in the agent's environment.
            org_level_connectors (List): A list of org level connector ids to be used by the agent.
            user_level_connectors (Dict): A dictionary mapping ApplicationConnectorType keys to lists of OAuth scopes. Each key represents a specific user level application connector, while the value is a list of scopes that define the permissions granted to the application.
            initialize_function_name (str): The name of the function to be used for initialization.
            initialize_function_code (str): The function code to be used for initialization.

        Returns:
            Agent: The new agent."""
        return self._call_api('createAgent', 'POST', query_params={}, body={'projectId': project_id, 'functionSourceCode': function_source_code, 'agentFunctionName': agent_function_name, 'name': name, 'memory': memory, 'packageRequirements': package_requirements, 'description': description, 'enableBinaryInput': enable_binary_input, 'evaluationFeatureGroupId': evaluation_feature_group_id, 'agentInputSchema': agent_input_schema, 'agentOutputSchema': agent_output_schema, 'workflowGraph': workflow_graph, 'agentInterface': agent_interface, 'includedModules': included_modules, 'orgLevelConnectors': org_level_connectors, 'userLevelConnectors': user_level_connectors, 'initializeFunctionName': initialize_function_name, 'initializeFunctionCode': initialize_function_code}, parse_type=Agent)

    def update_agent(self, model_id: str, function_source_code: str = None, agent_function_name: str = None, memory: int = None, package_requirements: list = None, description: str = None, enable_binary_input: bool = None, agent_input_schema: dict = None, agent_output_schema: dict = None, workflow_graph: Union[dict, WorkflowGraph] = None, agent_interface: Union[AgentInterface, str] = None, included_modules: List = None, org_level_connectors: List = None, user_level_connectors: Dict = None, initialize_function_name: str = None, initialize_function_code: str = None) -> Agent:
        """Updates an existing AI Agent. A new version of the agent will be created and published.

        Args:
            model_id (str): The unique ID associated with the AI Agent to be changed.
            memory (int): Memory (in GB) for the agent.
            package_requirements (list): A list of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0'].
            description (str): A description of the agent, including its purpose and instructions.
            workflow_graph (WorkflowGraph): The workflow graph for the agent.
            agent_interface (AgentInterface): The interface that the agent will be deployed with.
            included_modules (List): A list of user created custom modules to include in the agent's environment.
            org_level_connectors (List): A list of org level connector ids to be used by the agent.
            user_level_connectors (Dict): A dictionary mapping ApplicationConnectorType keys to lists of OAuth scopes. Each key represents a specific user level application connector, while the value is a list of scopes that define the permissions granted to the application.
            initialize_function_name (str): The name of the function to be used for initialization.
            initialize_function_code (str): The function code to be used for initialization.

        Returns:
            Agent: The updated agent."""
        return self._call_api('updateAgent', 'POST', query_params={}, body={'modelId': model_id, 'functionSourceCode': function_source_code, 'agentFunctionName': agent_function_name, 'memory': memory, 'packageRequirements': package_requirements, 'description': description, 'enableBinaryInput': enable_binary_input, 'agentInputSchema': agent_input_schema, 'agentOutputSchema': agent_output_schema, 'workflowGraph': workflow_graph, 'agentInterface': agent_interface, 'includedModules': included_modules, 'orgLevelConnectors': org_level_connectors, 'userLevelConnectors': user_level_connectors, 'initializeFunctionName': initialize_function_name, 'initializeFunctionCode': initialize_function_code}, parse_type=Agent)

    def generate_agent_code(self, project_id: str, prompt: str, fast_mode: bool = None) -> list:
        """Generates the code for defining an AI Agent

        Args:
            project_id (str): The unique ID associated with the project.
            prompt (str): A natural language prompt which describes agent specification. Describe what the agent will do, what inputs it will expect, and what outputs it will give out
            fast_mode (bool): If True, runs a faster but slightly less accurate code generation pipeline"""
        return self._call_api('generateAgentCode', 'POST', query_params={}, body={'projectId': project_id, 'prompt': prompt, 'fastMode': fast_mode})

    def evaluate_prompt(self, prompt: str = None, system_message: str = None, llm_name: Union[LLMName, str] = None, max_tokens: int = None, temperature: float = 0.0, messages: list = None, response_type: str = None, json_response_schema: dict = None, stop_sequences: List = None, top_p: float = None) -> LlmResponse:
        """Generate response to the prompt using the specified model.

        Args:
            prompt (str): Prompt to use for generation.
            system_message (str): System prompt for models that support it.
            llm_name (LLMName): Name of the underlying LLM to be used for generation. Default is auto selection.
            max_tokens (int): Maximum number of tokens to generate. If set, the model will just stop generating after this token limit is reached.
            temperature (float): Temperature to use for generation. Higher temperature makes more non-deterministic responses, a value of zero makes mostly deterministic reponses. Default is 0.0. A range of 0.0 - 2.0 is allowed.
            messages (list): A list of messages to use as conversation history. A message is a dict with attributes: is_user (bool): Whether the message is from the user. text (str): The message's text. attachments (list): The files attached to the message represented as a list of dictionaries [{"doc_id": <doc_id1>}, {"doc_id": <doc_id2>}]
            response_type (str): Specifies the type of response to request from the LLM. One of 'text' and 'json'. If set to 'json', the LLM will respond with a json formatted string whose schema can be specified `json_response_schema`. Defaults to 'text'
            json_response_schema (dict): A dictionary specifying the keys/schema/parameters which LLM should adhere to in its response when `response_type` is 'json'. Each parameter is mapped to a dict with the following info - type (str) (required): Data type of the parameter. description (str) (required): Description of the parameter. is_required (bool) (optional): Whether the parameter is required or not. Example: json_response_schema = {'title': {'type': 'string', 'description': 'Article title', 'is_required': true}, 'body': {'type': 'string', 'description': 'Article body'}}
            stop_sequences (List): Specifies the strings on which the LLM will stop generation.
            top_p (float): The nucleus sampling value used for this run. If set, the model will sample from the smallest set of tokens whose cumulative probability exceeds the probability `top_p`. Default is 1.0. A range of 0.0 - 1.0 is allowed. It is generally recommended to use either temperature sampling or nucleus sampling, but not both.

        Returns:
            LlmResponse: The response from the model, raw text and parsed components."""
        return self._proxy_request('EvaluatePrompt', 'POST', query_params={}, body={'prompt': prompt, 'systemMessage': system_message, 'llmName': llm_name, 'maxTokens': max_tokens, 'temperature': temperature, 'messages': messages, 'responseType': response_type, 'jsonResponseSchema': json_response_schema, 'stopSequences': stop_sequences, 'topP': top_p}, parse_type=LlmResponse)

    def render_feature_groups_for_llm(self, feature_group_ids: List, token_budget: int = None, include_definition: bool = True) -> List[LlmInput]:
        """Encode feature groups as language model inputs.

        Args:
            feature_group_ids (List): List of feature groups to be encoded.
            token_budget (int): Enforce a given budget for each encoded feature group.
            include_definition (bool): Include the definition of the feature group in the encoding.

        Returns:
            list[LlmInput]: LLM input object comprising of information about the feature groups with given IDs."""
        return self._call_api('renderFeatureGroupsForLLM', 'POST', query_params={}, body={'featureGroupIds': feature_group_ids, 'tokenBudget': token_budget, 'includeDefinition': include_definition}, parse_type=LlmInput)

    def generate_code_for_data_query_using_llm(self, query: str, feature_group_ids: List = None, external_database_schemas: List = None, prompt_context: str = None, llm_name: Union[LLMName, str] = None, temperature: float = None, sql_dialect: str = 'Spark') -> LlmGeneratedCode:
        """Execute a data query using a large language model in an async fashion.

        Args:
            query (str): The natural language query to execute. The query is converted to a SQL query using the language model.
            feature_group_ids (List): A list of feature group IDs that the query should be executed against.
            external_database_schemas (List): A list of schmeas from external database that the query should be executed against.
            prompt_context (str): The context message used to construct the prompt for the language model. If not provide, a default context message is used.
            llm_name (LLMName): The name of the language model to use. If not provided, the default language model is used.
            temperature (float): The temperature to use for the language model if supported. If not provided, the default temperature is used.
            sql_dialect (str): The dialect of sql to generate sql for. The default is Spark.

        Returns:
            LlmGeneratedCode: The generated SQL code."""
        return self._proxy_request('GenerateCodeForDataQueryUsingLlm', 'POST', query_params={}, body={'query': query, 'featureGroupIds': feature_group_ids, 'externalDatabaseSchemas': external_database_schemas, 'promptContext': prompt_context, 'llmName': llm_name, 'temperature': temperature, 'sqlDialect': sql_dialect}, parse_type=LlmGeneratedCode)

    def extract_data_using_llm(self, field_descriptors: List, document_id: str = None, document_text: str = None, llm_name: Union[LLMName, str] = None) -> ExtractedFields:
        """Extract fields from a document using a large language model.

        Args:
            field_descriptors (List): A list of fields to extract from the document.
            document_id (str): The ID of the document to query.
            document_text (str): The text of the document to query. Only used if document_id is not provided.
            llm_name (LLMName): The name of the language model to use. If not provided, the default language model is used.

        Returns:
            ExtractedFields: The response from the document query."""
        return self._proxy_request('ExtractDataUsingLlm', 'POST', query_params={}, body={'fieldDescriptors': field_descriptors, 'documentId': document_id, 'documentText': document_text, 'llmName': llm_name}, parse_type=ExtractedFields)

    def search_web_for_llm(self, queries: List, search_providers: List = None, max_results: int = 1, safe: bool = True, fetch_content: bool = False, max_page_tokens: int = 8192, convert_to_markdown: bool = True) -> WebSearchResponse:
        """Access web search providers to fetch content related to the queries for use in large language model inputs.

        This method can access multiple search providers and return information from them. If the provider supplies
        URLs for the results then this method also supports fetching the contents of those URLs, optionally converting
        them to markdown format, and returning them as part of the response. Set a token budget to limit the amount of
        content returned in the response.


        Args:
            queries (List): List of queries to send to the search providers. At most 10 queries each less than 512 characters.
            search_providers (List): Search providers to use for the search. If not provided a default provider is used. - BING - GOOGLE
            max_results (int): Maximum number of results to fetch per provider. Must be in [1, 100]. Defaults to 1 (I'm feeling lucky).
            safe (bool): Whether content safety is enabled for these search request. Defaults to True.
            fetch_content (bool): If true fetches the content from the urls in the search results. Defailts to False.
            max_page_tokens (int): Maximum number of tokens to accumulate if fetching search result contents.
            convert_to_markdown (bool): Whether content should be converted to markdown. Defaults to True.

        Returns:
            WebSearchResponse: Results of running the search queries."""
        return self._proxy_request('SearchWebForLlm', 'POST', query_params={}, body={'queries': queries, 'searchProviders': search_providers, 'maxResults': max_results, 'safe': safe, 'fetchContent': fetch_content, 'maxPageTokens': max_page_tokens, 'convertToMarkdown': convert_to_markdown}, parse_type=WebSearchResponse)

    def fetch_web_page(self, url: str, convert_to_markdown: bool = True) -> WebPageResponse:
        """Scrapes the content of a web page and returns it as a string.

        Args:
            url (str): The url of the web page to scrape.
            convert_to_markdown (bool): Whether content should be converted to markdown.

        Returns:
            WebPageResponse: The content of the web page."""
        return self._proxy_request('FetchWebPage', 'POST', query_params={}, body={'url': url, 'convertToMarkdown': convert_to_markdown}, parse_type=WebPageResponse)

    def construct_agent_conversation_messages_for_llm(self, deployment_conversation_id: str = None, external_session_id: str = None, include_document_contents: bool = True) -> AgentConversation:
        """Returns conversation history in a format for LLM calls.

        Args:
            deployment_conversation_id (str): Unique ID of the conversation. One of deployment_conversation_id or external_session_id must be provided.
            external_session_id (str): External session ID of the conversation.
            include_document_contents (bool): If true, include contents from uploaded documents in the generated messages.

        Returns:
            AgentConversation: Contains a list of AgentConversationMessage that represents the conversation."""
        return self._proxy_request('constructAgentConversationMessagesForLLM', 'POST', query_params={}, body={'deploymentConversationId': deployment_conversation_id, 'externalSessionId': external_session_id, 'includeDocumentContents': include_document_contents}, parse_type=AgentConversation, is_sync=True)

    def validate_workflow_graph(self, workflow_graph: Union[dict, WorkflowGraph], agent_interface: Union[AgentInterface, str] = AgentInterface.DEFAULT, package_requirements: list = []) -> dict:
        """Validates the workflow graph for an AI Agent.

        Args:
            workflow_graph (WorkflowGraph): The workflow graph to validate.
            agent_interface (AgentInterface): The interface that the agent will be deployed with.
            package_requirements (list): A list of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']."""
        return self._call_api('validateWorkflowGraph', 'POST', query_params={}, body={'workflowGraph': workflow_graph, 'agentInterface': agent_interface, 'packageRequirements': package_requirements})

    def extract_agent_workflow_information(self, workflow_graph: Union[dict, WorkflowGraph], agent_interface: Union[AgentInterface, str] = AgentInterface.DEFAULT, package_requirements: list = []) -> dict:
        """Extracts source code of workflow graph, ancestors, in_edges and traversal orders from the agent workflow.

        Args:
            workflow_graph (WorkflowGraph): The workflow graph to validate.
            agent_interface (AgentInterface): The interface that the agent will be deployed with.
            package_requirements (list): A list of package requirement strings. For example: ['numpy==1.2.3', 'pandas>=1.4.0']."""
        return self._call_api('extractAgentWorkflowInformation', 'POST', query_params={}, body={'workflowGraph': workflow_graph, 'agentInterface': agent_interface, 'packageRequirements': package_requirements})

    def get_llm_app_response(self, llm_app_name: str, prompt: str) -> LlmResponse:
        """Queries the specified LLM App to generate a response to the prompt. LLM Apps are LLMs tailored to achieve a specific task like code generation for a specific service's API.

        Args:
            llm_app_name (str): The name of the LLM App to use for generation.
            prompt (str): The prompt to use for generation.

        Returns:
            LlmResponse: The response from the LLM App."""
        return self._call_api('getLLMAppResponse', 'POST', query_params={}, body={'llmAppName': llm_app_name, 'prompt': prompt}, parse_type=LlmResponse)

    def create_document_retriever(self, project_id: str, name: str, feature_group_id: str, document_retriever_config: Union[dict, VectorStoreConfig] = None) -> DocumentRetriever:
        """Returns a document retriever that stores embeddings for document chunks in a feature group.

        Document columns in the feature group are broken into chunks. For cases with multiple document columns, chunks from all columns are combined together to form a single chunk.


        Args:
            project_id (str): The ID of project that the Document Retriever is created in.
            name (str): The name of the Document Retriever. Can be up to 120 characters long and can only contain alphanumeric characters and underscores.
            feature_group_id (str): The ID of the feature group that the Document Retriever is associated with.
            document_retriever_config (VectorStoreConfig): The configuration, including chunk_size and chunk_overlap_fraction, for document retrieval.

        Returns:
            DocumentRetriever: The newly created document retriever."""
        return self._call_api('createDocumentRetriever', 'POST', query_params={}, body={'projectId': project_id, 'name': name, 'featureGroupId': feature_group_id, 'documentRetrieverConfig': document_retriever_config}, parse_type=DocumentRetriever)

    def rename_document_retriever(self, document_retriever_id: str, name: str) -> DocumentRetriever:
        """Updates an existing document retriever.

        Args:
            document_retriever_id (str): The unique ID associated with the document retriever.
            name (str): The name to update the document retriever with.

        Returns:
            DocumentRetriever: The updated document retriever."""
        return self._call_api('renameDocumentRetriever', 'POST', query_params={}, body={'documentRetrieverId': document_retriever_id, 'name': name}, parse_type=DocumentRetriever)

    def create_document_retriever_version(self, document_retriever_id: str, feature_group_id: str = None, document_retriever_config: Union[dict, VectorStoreConfig] = None) -> DocumentRetrieverVersion:
        """Creates a document retriever version from the latest version of the feature group that the document retriever associated with.

        Args:
            document_retriever_id (str): The unique ID associated with the document retriever to create version with.
            feature_group_id (str): The ID of the feature group to update the document retriever with.
            document_retriever_config (VectorStoreConfig): The configuration, including chunk_size and chunk_overlap_fraction, for document retrieval.

        Returns:
            DocumentRetrieverVersion: The newly created document retriever version."""
        return self._call_api('createDocumentRetrieverVersion', 'POST', query_params={}, body={'documentRetrieverId': document_retriever_id, 'featureGroupId': feature_group_id, 'documentRetrieverConfig': document_retriever_config}, parse_type=DocumentRetrieverVersion)

    def delete_document_retriever(self, vector_store_id: str):
        """Delete a Document Retriever.

        Args:
            vector_store_id (str): A unique string identifier associated with the document retriever."""
        return self._call_api('deleteDocumentRetriever', 'DELETE', query_params={'vectorStoreId': vector_store_id})

    def delete_document_retriever_version(self, document_retriever_version: str):
        """Delete a document retriever version.

        Args:
            document_retriever_version (str): A unique string identifier associated with the document retriever version."""
        return self._call_api('deleteDocumentRetrieverVersion', 'DELETE', query_params={'documentRetrieverVersion': document_retriever_version})

    def get_document_snippet(self, document_retriever_id: str, document_id: str, start_word_index: int = None, end_word_index: int = None) -> DocumentRetrieverLookupResult:
        """Get a snippet from documents in the document retriever.

        Args:
            document_retriever_id (str): A unique string identifier associated with the document retriever.
            document_id (str): The ID of the document to retrieve the snippet from.
            start_word_index (int): If provided, will start the snippet at the index (of words in the document) specified.
            end_word_index (int): If provided, will end the snippet at the index of (of words in the document) specified.

        Returns:
            DocumentRetrieverLookupResult: The documentation snippet found from the document retriever."""
        return self._call_api('getDocumentSnippet', 'POST', query_params={}, body={'documentRetrieverId': document_retriever_id, 'documentId': document_id, 'startWordIndex': start_word_index, 'endWordIndex': end_word_index}, parse_type=DocumentRetrieverLookupResult)

    def restart_document_retriever(self, document_retriever_id: str):
        """Restart the document retriever if it is stopped or has failed. This will start the deployment of the document retriever,

        but will not wait for it to be ready. You need to call wait_until_ready to wait until the deployment is ready.


        Args:
            document_retriever_id (str): A unique string identifier associated with the document retriever."""
        return self._call_api('restartDocumentRetriever', 'POST', query_params={}, body={'documentRetrieverId': document_retriever_id})

    def get_relevant_snippets(self, doc_ids: List = None, blobs: io.TextIOBase = None, query: str = None, document_retriever_config: Union[dict, VectorStoreConfig] = None, honor_sentence_boundary: bool = True, num_retrieval_margin_words: int = None, max_words_per_snippet: int = None, max_snippets_per_document: int = None, start_word_index: int = None, end_word_index: int = None, including_bounding_boxes: bool = False, text: str = None, document_processing_config: Union[dict, DocumentProcessingConfig] = None) -> List[DocumentRetrieverLookupResult]:
        """Retrieves snippets relevant to a given query from specified documents. This function supports flexible input options,

        allowing for retrieval from a variety of data sources including document IDs, blob data, and plain text. When multiple data
        sources are provided, all are considered in the retrieval process. Document retrievers may be created on-the-fly to perform lookup.


        Args:
            doc_ids (List): A list of document store IDs to retrieve the snippets from.
            blobs (io.TextIOBase): A dictionary mapping document names to the blob data.
            query (str): Query string to find relevant snippets in the documents.
            document_retriever_config (VectorStoreConfig): If provided, used to configure the retrieval steps like chunking for embeddings.
            num_retrieval_margin_words (int): If provided, will add this number of words from left and right of the returned snippets.
            max_words_per_snippet (int): If provided, will limit the number of words in each snippet to the value specified.
            max_snippets_per_document (int): If provided, will limit the number of snippets retrieved from each document to the value specified.
            start_word_index (int): If provided, will start the snippet at the index (of words in the document) specified.
            end_word_index (int): If provided, will end the snippet at the index of (of words in the document) specified.
            including_bounding_boxes (bool): If true, will include the bounding boxes of the snippets if they are available.
            text (str): Plain text from which to retrieve snippets.
            document_processing_config (DocumentProcessingConfig): The document processing configuration used to extract text when doc_ids or blobs are provided. If provided, this will override including_bounding_boxes parameter.

        Returns:
            list[DocumentRetrieverLookupResult]: The snippets found from the documents."""
        return self._proxy_request('GetRelevantSnippets', 'POST', query_params={}, data={'docIds': doc_ids, 'query': query, 'documentRetrieverConfig': json.dumps(document_retriever_config.to_dict()) if hasattr(document_retriever_config, 'to_dict') else json.dumps(document_retriever_config), 'honorSentenceBoundary': honor_sentence_boundary, 'numRetrievalMarginWords': num_retrieval_margin_words, 'maxWordsPerSnippet': max_words_per_snippet, 'maxSnippetsPerDocument': max_snippets_per_document, 'startWordIndex': start_word_index, 'endWordIndex': end_word_index, 'includingBoundingBoxes': including_bounding_boxes, 'text': text, 'documentProcessingConfig': json.dumps(document_processing_config.to_dict()) if hasattr(document_processing_config, 'to_dict') else json.dumps(document_processing_config)}, files=blobs, parse_type=DocumentRetrieverLookupResult)
