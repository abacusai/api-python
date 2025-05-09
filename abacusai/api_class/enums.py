from enum import Enum
from typing import Union


def deprecated_enums(*enum_values):
    def enum_class_wrapper(cls):
        cls.__deprecated_values__ = list(enum_values)
        return cls
    return enum_class_wrapper


class ApiEnum(Enum):
    __deprecated_values__ = []

    def is_deprecated(self):
        return self.value in self.__deprecated_values__

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value.upper() == other.upper()
        elif isinstance(other, int):
            return self.value == other
        elif other is None:
            return self.value == ''
        return super().__eq__(other)

    def __hash__(self):
        if isinstance(self.value, str):
            return hash(self.value.upper())
        return hash(self.value)


class ProblemType(ApiEnum):
    AI_AGENT = 'ai_agent'
    EVENT_ANOMALY = 'event_anomaly'
    CLUSTERING = 'clustering'
    CLUSTERING_TIMESERIES = 'clustering_timeseries'
    CUMULATIVE_FORECASTING = 'cumulative_forecasting'
    NAMED_ENTITY_EXTRACTION = 'nlp_ner'
    NATURAL_LANGUAGE_SEARCH = 'nlp_search'
    CHAT_LLM = 'chat_llm'
    SENTENCE_BOUNDARY_DETECTION = 'nlp_sentence_boundary_detection'
    SENTIMENT_DETECTION = 'nlp_sentiment'
    DOCUMENT_CLASSIFICATION = 'nlp_classification'
    DOCUMENT_SUMMARIZATION = 'nlp_summarization'
    DOCUMENT_VISUALIZATION = 'nlp_document_visualization'
    PERSONALIZATION = 'personalization'
    PREDICTIVE_MODELING = 'regression'
    FINETUNED_LLM = 'finetuned_llm'
    FORECASTING = 'forecasting'
    CUSTOM_TRAINED_MODEL = 'plug_and_play'
    CUSTOM_ALGORITHM = 'trainable_plug_and_play'
    FEATURE_STORE = 'feature_store'
    IMAGE_CLASSIFICATION = 'vision_classification'
    OBJECT_DETECTION = 'vision_object_detection'
    IMAGE_VALUE_PREDICTION = 'vision_regression'
    MODEL_MONITORING = 'model_monitoring'
    LANGUAGE_DETECTION = 'language_detection'
    OPTIMIZATION = 'optimization'
    PRETRAINED_MODELS = 'pretrained'
    THEME_ANALYSIS = 'theme_analysis'
    TS_ANOMALY = 'ts_anomaly'


class RegressionObjective(ApiEnum):

    AUC = 'auc'
    ACCURACY = 'acc'
    LOG_LOSS = 'log_loss'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1_SCORE = 'fscore'
    MAE = 'mae'
    MAPE = 'mape'
    WAPE = 'wape'
    RMSE = 'rmse'
    R_SQUARED_COEFFICIENT_OF_DETERMINATION = 'r^2'


class RegressionTreeHPOMode(ApiEnum):
    RAPID = 'rapid'
    THOROUGH = 'thorough'


class PartialDependenceAnalysis(ApiEnum):
    RAPID = 'rapid'
    THOROUGH = 'thorough'


class RegressionAugmentationStrategy(ApiEnum):
    SMOTE = 'smote'
    RESAMPLE = 'resample'


class RegressionTargetTransform(ApiEnum):
    LOG = 'log'
    QUANTILE = 'quantile'
    YEO_JOHNSON = 'yeo-johnson'
    BOX_COX = 'box-cox'


class RegressionTypeOfSplit(ApiEnum):
    RANDOM = 'Random Sampling'
    TIMESTAMP_BASED = 'Timestamp Based'
    ROW_INDICATOR_BASED = 'Row Indicator Based'
    STRATIFIED_RANDOM_SAMPLING = 'Stratified Random Sampling'


class RegressionTimeSplitMethod(ApiEnum):
    TEST_SPLIT_PERCENTAGE_BASED = 'Test Split Percentage Based'
    TEST_START_TIMESTAMP_BASED = 'Test Start Timestamp Based'


class RegressionLossFunction(ApiEnum):
    HUBER = 'Huber'
    MSE = 'Mean Squared Error'
    MAE = 'Mean Absolute Error'
    MAPE = 'Mean Absolute Percentage Error'
    MSLE = 'Mean Squared Logarithmic Error'
    TWEEDIE = 'Tweedie'
    CROSS_ENTROPY = 'Cross Entropy'
    FOCAL_CROSS_ENTROPY = 'Focal Cross Entropy'
    AUTOMATIC = 'Automatic'
    CUSTOM = 'Custom'


class ExplainerType(ApiEnum):
    KERNEL_EXPLAINER = 'KERNEL_EXPLAINER'
    LIME_EXPLAINER = 'LIME_EXPLAINER'
    TREE_EXPLAINER = 'TREE_EXPLAINER'
    EBM_EXPLAINER = 'EBM_EXPLAINER'


class SamplingMethodType(ApiEnum):
    N_SAMPLING = 'N_SAMPLING'
    PERCENT_SAMPLING = 'PERCENT_SAMPLING'


class MergeMode(ApiEnum):
    LAST_N = 'LAST_N'
    TIME_WINDOW = 'TIME_WINDOW'


class OperatorType(ApiEnum):
    UNPIVOT = 'UNPIVOT'
    MARKDOWN = 'MARKDOWN'
    CRAWLER = 'CRAWLER'
    EXTRACT_DOCUMENT_DATA = 'EXTRACT_DOCUMENT_DATA'
    DATA_GENERATION = 'DATA_GENERATION'
    UNION = 'UNION'


class MarkdownOperatorInputType(ApiEnum):
    HTML = 'HTML'


class FillLogic(ApiEnum):
    # back / future
    AVERAGE = 'average'
    MAX = 'max'
    MEDIAN = 'median'
    MIN = 'min'
    CUSTOM = 'custom'
    # middle
    BACKFILL = 'bfill'
    FORWARDFILL = 'ffill'
    LINEAR = 'linear'
    NEAREST = 'nearest'


class BatchSize(ApiEnum):
    BATCH_8 = 8
    BATCH_16 = 16
    BATCH_32 = 32
    BATCH_64 = 64
    BATCH_128 = 128
    BATCH_256 = 256
    BATCH_384 = 384
    BATCH_512 = 512
    BATCH_740 = 740
    BATCH_1024 = 1024


class HolidayCalendars(ApiEnum):
    AU = 'AU'
    UK = 'UK'
    US = 'US'


class FileFormat(ApiEnum):
    AVRO = 'AVRO'
    PARQUET = 'PARQUET'
    TFRECORD = 'TFRECORD'
    TSV = 'TSV'
    CSV = 'CSV'
    ORC = 'ORC'
    JSON = 'JSON'
    ODS = 'ODS'
    XLS = 'XLS'
    GZ = 'GZ'
    ZIP = 'ZIP'
    TAR = 'TAR'
    DOCX = 'DOCX'
    PDF = 'PDF'
    MD = 'md'
    RAR = 'RAR'
    GIF = 'GIF'
    JPEG = 'JPG'
    PNG = 'PNG'
    TIF = 'TIFF'
    NUMBERS = 'NUMBERS'
    PPTX = 'PPTX'
    PPT = 'PPT'
    HTML = 'HTML'
    TXT = 'txt'
    EML = 'eml'
    MP3 = 'MP3'
    MP4 = 'MP4'
    FLV = 'flv'
    MOV = 'mov'
    MPG = 'mpg'
    MPEG = 'mpeg'
    WEBP = 'webp'
    WEBM = 'webm'
    WMV = 'wmv'
    MSG = 'msg'


class ExperimentationMode(ApiEnum):
    RAPID = 'rapid'
    THOROUGH = 'thorough'


class PersonalizationTrainingMode(ApiEnum):
    EXPERIMENTAL = 'EXP'
    PRODUCTION = 'PROD'


class PersonalizationObjective(ApiEnum):
    NDCG = 'ndcg'
    NDCG_5 = 'ndcg@5'
    NDCG_10 = 'ndcg@10'
    MAP = 'map'
    MAP_5 = 'map@5'
    MAP_10 = 'map@10'
    MRR = 'mrr'
    PERSONALIZATION = 'personalization@10'
    COVERAGE = 'coverage'


# Forecasting
class ForecastingObjective(ApiEnum):
    ACCURACY = 'w_c_accuracy'
    WAPE = 'wape'
    MAPE = 'mape'
    CMAPE = 'cmape'
    RMSE = 'rmse'
    CV = 'coefficient_of_variation'
    BIAS = 'bias'
    SRMSE = 'srmse'


class ForecastingFrequency(ApiEnum):
    HOURLY = '1H'
    DAILY = '1D'
    WEEKLY_SUNDAY_START = '1W'
    WEEKLY_MONDAY_START = 'W-MON'
    WEEKLY_SATURDAY_START = 'W-SAT'
    MONTH_START = 'MS'
    MONTH_END = '1M'
    QUARTER_START = 'QS'
    QUARTER_END = '1Q'
    YEARLY = '1Y'


class ForecastingDataSplitType(ApiEnum):
    AUTO = 'Automatic Time Based'
    TIMESTAMP = 'Timestamp Based'
    ITEM = 'Item Based'
    PREDICTION_LENGTH = 'Force Prediction Length'
    L_SHAPED_AUTO = 'L-shaped Split - Automatic Time Based'
    L_SHAPED_TIMESTAMP = 'L-shaped Split - Timestamp Based'


class ForecastingLossFunction(ApiEnum):
    CUSTOM = 'Custom'
    MEAN_ABSOLUTE_ERROR = 'mae'
    NORMALIZED_MEAN_ABSOLUTE_ERROR = 'nmae'
    PEAKS_MEAN_ABSOLUTE_ERROR = 'peaks_mae'
    MEAN_ABSOLUTE_PERCENTAGE_ERROR = 'stable_mape'
    POINTWISE_ACCURACY = 'accuracy'
    ROOT_MEAN_SQUARE_ERROR = 'rmse'
    NORMALIZED_ROOT_MEAN_SQUARE_ERROR = 'nrmse'
    ASYMMETRIC_MEAN_ABSOLUTE_PERCENTAGE_ERROR = 'asymmetric_mape'
    STABLE_STANDARDIZED_MEAN_ABSOLUTE_PERCENTAGE_ERROR = 'stable_standardized_mape_with_cmape'
    GAUSSIAN = 'mle_gaussian_local'
    GAUSSIAN_FULL_COVARIANCE = 'mle_gaussfullcov'
    GUASSIAN_EXPONENTIAL = 'mle_gaussexp'
    MIX_GAUSSIANS = 'mle_gaussmix'
    WEIBULL = 'mle_weibull'
    NEGATIVE_BINOMIAL = 'mle_negbinom'
    LOG_ROOT_MEAN_SQUARE_ERROR = 'log_rmse'


class ForecastingLocalScaling(ApiEnum):
    ZSCORE = 'zscore'
    SLIDING_ZSCORE = 'sliding_zscore'
    LAST_POINT = 'lastpoint'
    MIN_MAX = 'minmax'
    MIN_STD = 'minstd'
    ROBUST = 'robust'
    ITEM = 'item'


class ForecastingFillMethod(ApiEnum):
    BACK = 'BACK'
    MIDDLE = 'MIDDLE'
    FUTURE = 'FUTURE'


class ForecastingQuanitlesExtensionMethod(ApiEnum):
    DIRECT = 'direct'
    QUADRATIC = 'quadratic'
    ANCESTRAL_SIMULATION = 'simulation'


# Timeseries Anomaly Detection
class TimeseriesAnomalyDataSplitType(ApiEnum):
    AUTO = 'Automatic Time Based'
    TIMESTAMP = 'Fixed Timestamp Based'


class TimeseriesAnomalyTypeOfAnomaly(ApiEnum):
    HIGH_PEAK = 'high_peak'
    LOW_PEAK = 'low_peak'


class TimeseriesAnomalyUseHeuristic(ApiEnum):
    ENABLE = 'enable'
    DISABLE = 'disable'
    AUTOMATIC = 'automatic'


# Named Entity Recognition
class NERObjective(ApiEnum):
    LOG_LOSS = 'log_loss'
    AUC = 'auc'
    PRECISION = 'precision'
    RECALL = 'recall'
    ANNOTATIONS_PRECISION = 'annotations_precision'
    ANNOTATIONS_RECALL = 'annotations_recall'


# Named Entity Recognition


class NERModelType(ApiEnum):
    PRETRAINED_BERT = 'pretrained_bert'
    PRETRAINED_ROBERTA_27 = 'pretrained_roberta_27'
    PRETRAINED_ROBERTA_43 = 'pretrained_roberta_43'
    PRETRAINED_MULTILINGUAL = 'pretrained_multilingual'
    LEARNED = 'learned'


class NLPDocumentFormat(ApiEnum):
    AUTO = 'auto'
    TEXT = 'text'
    DOC = 'doc'
    TOKENS = 'tokens'


# Sentiment Analysis
class SentimentType(ApiEnum):
    VALENCE = 'valence'
    EMOTION = 'emotion'


# Timeseries Clustering
class ClusteringImputationMethod(ApiEnum):
    AUTOMATIC = 'Automatic'
    ZEROS = 'Zeros'
    INTERPOLATE = 'Interpolate'


class ConnectorType(ApiEnum):
    FILE = 'FILE'
    DATABASE = 'DATABASE'
    STREAMING = 'STREAMING'
    APPLICATION = 'APPLICATION'


class ApplicationConnectorType(ApiEnum):
    GOOGLEANALYTICS = 'GOOGLEANALYTICS'
    GOOGLEDRIVE = 'GOOGLEDRIVE'
    GOOGLECALENDAR = 'GOOGLECALENDAR'
    GIT = 'GIT'
    CONFLUENCE = 'CONFLUENCE'
    JIRA = 'JIRA'
    ONEDRIVE = 'ONEDRIVE'
    ZENDESK = 'ZENDESK'
    SLACK = 'SLACK'
    SHAREPOINT = 'SHAREPOINT'
    TEAMS = 'TEAMS'
    ABACUSUSAGEMETRICS = 'ABACUSUSAGEMETRICS'
    MICROSOFTAUTH = 'MICROSOFTAUTH'
    FRESHSERVICE = 'FRESHSERVICE'
    ZENDESKSUNSHINEMESSAGING = 'ZENDESKSUNSHINEMESSAGING'
    GOOGLEDRIVEUSER = 'GOOGLEDRIVEUSER'
    GOOGLEWORKSPACEUSER = 'GOOGLEWORKSPACEUSER'
    GMAILUSER = 'GMAILUSER'
    GOOGLESHEETS = 'GOOGLESHEETS'
    GOOGLEDOCS = 'GOOGLEDOCS'
    TEAMSSCRAPER = 'TEAMSSCRAPER'
    GITHUBUSER = 'GITHUBUSER'
    OKTASAML = 'OKTASAML'
    BOX = 'BOX'
    SFTPAPPLICATION = 'SFTPAPPLICATION'
    OAUTH = 'OAUTH'


class StreamingConnectorType(ApiEnum):
    KAFKA = 'KAFKA'


class PythonFunctionArgumentType(ApiEnum):
    FEATURE_GROUP = 'FEATURE_GROUP'
    INTEGER = 'INTEGER'
    STRING = 'STRING'
    BOOLEAN = 'BOOLEAN'
    FLOAT = 'FLOAT'
    JSON = 'JSON'
    LIST = 'LIST'
    DATASET_ID = 'DATASET_ID'
    MODEL_ID = 'MODEL_ID'
    FEATURE_GROUP_ID = 'FEATURE_GROUP_ID'
    MONITOR_ID = 'MONITOR_ID'
    BATCH_PREDICTION_ID = 'BATCH_PREDICTION_ID'
    DEPLOYMENT_ID = 'DEPLOYMENT_ID'
    ATTACHMENT = 'ATTACHMENT'

    @staticmethod
    def to_json_type(type):
        if type == PythonFunctionArgumentType.INTEGER or type == PythonFunctionArgumentType.FLOAT:
            return 'number'
        elif type == PythonFunctionArgumentType.STRING or type == PythonFunctionArgumentType.ATTACHMENT:
            return 'string'
        elif type == PythonFunctionArgumentType.BOOLEAN:
            return 'boolean'
        elif type == PythonFunctionArgumentType.LIST:
            return 'array'
        elif type == PythonFunctionArgumentType.JSON:
            return 'object'
        else:
            raise ValueError(f'Invalid type: {type}. type not JSON compatible')


class PythonFunctionOutputArgumentType(ApiEnum):
    NTEGER = 'INTEGER'
    STRING = 'STRING'
    BOOLEAN = 'BOOLEAN'
    FLOAT = 'FLOAT'
    JSON = 'JSON'
    LIST = 'LIST'
    DATASET_ID = 'DATASET_ID'
    MODEL_ID = 'MODEL_ID'
    FEATURE_GROUP_ID = 'FEATURE_GROUP_ID'
    MONITOR_ID = 'MONITOR_ID'
    BATCH_PREDICTION_ID = 'BATCH_PREDICTION_ID'
    DEPLOYMENT_ID = 'DEPLOYMENT_ID'
    ANY = 'ANY'
    ATTACHMENT = 'ATTACHMENT'


class VectorStoreTextEncoder(ApiEnum):
    E5 = 'E5'
    OPENAI = 'OPENAI'
    OPENAI_COMPACT = 'OPENAI_COMPACT'
    OPENAI_LARGE = 'OPENAI_LARGE'
    SENTENCE_BERT = 'SENTENCE_BERT'
    E5_SMALL = 'E5_SMALL'
    CODE_BERT = 'CODE_BERT'


@deprecated_enums('OPENAI_GPT4_32K', 'OPENAI_GPT3_5', 'OPENAI_GPT3_5_TEXT',
                  'OPENAI_GPT4', 'OPENAI_GPT4_128K', 'OPENAI_GPT4_128K_LATEST',
                  'LLAMA3_LARGE_CHAT', 'CLAUDE_V3_OPUS', 'CLAUDE_V3_SONNET', 'CLAUDE_V3_HAIKU',
                  'QWEN_2_5_32B_BASE')
class LLMName(ApiEnum):
    OPENAI_GPT4 = 'OPENAI_GPT4'
    OPENAI_GPT4_32K = 'OPENAI_GPT4_32K'
    OPENAI_GPT4_128K = 'OPENAI_GPT4_128K'
    OPENAI_GPT4_128K_LATEST = 'OPENAI_GPT4_128K_LATEST'
    OPENAI_GPT4O = 'OPENAI_GPT4O'
    OPENAI_GPT4O_MINI = 'OPENAI_GPT4O_MINI'
    OPENAI_O1_MINI = 'OPENAI_O1_MINI'
    OPENAI_GPT4_1 = 'OPENAI_GPT4_1'
    OPENAI_GPT4_1_MINI = 'OPENAI_GPT4_1_MINI'
    OPENAI_GPT3_5 = 'OPENAI_GPT3_5'
    OPENAI_GPT3_5_TEXT = 'OPENAI_GPT3_5_TEXT'
    LLAMA3_1_405B = 'LLAMA3_1_405B'
    LLAMA3_1_70B = 'LLAMA3_1_70B'
    LLAMA3_1_8B = 'LLAMA3_1_8B'
    LLAMA3_3_70B = 'LLAMA3_3_70B'
    LLAMA3_LARGE_CHAT = 'LLAMA3_LARGE_CHAT'
    CLAUDE_V3_OPUS = 'CLAUDE_V3_OPUS'
    CLAUDE_V3_SONNET = 'CLAUDE_V3_SONNET'
    CLAUDE_V3_HAIKU = 'CLAUDE_V3_HAIKU'
    CLAUDE_V3_5_SONNET = 'CLAUDE_V3_5_SONNET'
    CLAUDE_V3_7_SONNET = 'CLAUDE_V3_7_SONNET'
    CLAUDE_V3_5_HAIKU = 'CLAUDE_V3_5_HAIKU'
    GEMINI_1_5_PRO = 'GEMINI_1_5_PRO'
    GEMINI_2_FLASH = 'GEMINI_2_FLASH'
    GEMINI_2_FLASH_THINKING = 'GEMINI_2_FLASH_THINKING'
    GEMINI_2_PRO = 'GEMINI_2_PRO'
    ABACUS_SMAUG3 = 'ABACUS_SMAUG3'
    ABACUS_DRACARYS = 'ABACUS_DRACARYS'
    QWEN_2_5_32B = 'QWEN_2_5_32B'
    QWEN_2_5_32B_BASE = 'QWEN_2_5_32B_BASE'
    QWEN_2_5_72B = 'QWEN_2_5_72B'
    QWQ_32B = 'QWQ_32B'
    QWEN3_235B_A22B = 'QWEN3_235B_A22B'
    GEMINI_1_5_FLASH = 'GEMINI_1_5_FLASH'
    XAI_GROK = 'XAI_GROK'
    XAI_GROK_3 = 'XAI_GROK_3'
    XAI_GROK_3_MINI = 'XAI_GROK_3_MINI'
    DEEPSEEK_V3 = 'DEEPSEEK_V3'
    DEEPSEEK_R1 = 'DEEPSEEK_R1'


class MonitorAlertType(ApiEnum):
    ACCURACY_BELOW_THRESHOLD = 'AccuracyBelowThreshold'
    FEATURE_DRIFT = 'FeatureDrift'
    DATA_INTEGRITY_VIOLATIONS = 'DataIntegrityViolations'
    BIAS_VIOLATIONS = 'BiasViolations'
    HISTORY_LENGTH_DRIFT = 'HistoryLengthDrift'
    TARGET_DRIFT = 'TargetDrift'
    PREDICTION_COUNT = 'PredictionCount'


class FeatureDriftType(ApiEnum):
    KL = 'kl'
    KS = 'ks'
    WS = 'ws'
    JS = 'js'
    PSI = 'psi'
    CHI_SQUARE = 'chi_square'
    CSI = 'csi'


class DataIntegrityViolationType(ApiEnum):
    NULL_VIOLATIONS = 'null_violations'
    RANGE_VIOLATIONS = 'range_violations'
    CATEGORICAL_RANGE_VIOLATION = 'categorical_range_violations'
    TOTAL_VIOLATIONS = 'total_violations'


class BiasType(ApiEnum):
    DEMOGRAPHIC_PARITY = 'demographic_parity'
    EQUAL_OPPORTUNITY = 'equal_opportunity'
    GROUP_BENEFIT_EQUALITY = 'group_benefit'
    TOTAL = 'total'


class AlertActionType(ApiEnum):
    EMAIL = 'Email'


class PythonFunctionType(ApiEnum):
    FEATURE_GROUP = 'FEATURE_GROUP'
    PLOTLY_FIG = 'PLOTLY_FIG'
    STEP_FUNCTION = 'STEP_FUNCTION'
    USERCODE_TOOL = 'USERCODE_TOOL'
    CONNECTOR_TOOL = 'CONNECTOR_TOOL'


class EvalArtifactType(ApiEnum):
    FORECASTING_ACCURACY = 'bar_chart'
    FORECASTING_VOLUME = 'bar_chart_volume'
    FORECASTING_HISTORY_LENGTH_ACCURACY = 'bar_chart_accuracy_by_history'


class FieldDescriptorType(ApiEnum):
    STRING = 'STRING'
    INTEGER = 'INTEGER'
    FLOAT = 'FLOAT'
    BOOLEAN = 'BOOLEAN'
    DATETIME = 'DATETIME'
    DATE = 'DATE'


class WorkflowNodeInputType(ApiEnum):
    # Duplicated in reainternal.enums, both should be kept in sync
    USER_INPUT = 'USER_INPUT'
    WORKFLOW_VARIABLE = 'WORKFLOW_VARIABLE'
    IGNORE = 'IGNORE'
    CONSTANT = 'CONSTANT'


class WorkflowNodeOutputType(ApiEnum):
    ATTACHMENT = 'ATTACHMENT'
    BOOLEAN = 'BOOLEAN'
    FLOAT = 'FLOAT'
    INTEGER = 'INTEGER'
    DICT = 'DICT'
    LIST = 'LIST'
    STRING = 'STRING'
    RUNTIME_SCHEMA = 'RUNTIME_SCHEMA'
    ANY = 'ANY'

    @classmethod
    def normalize_type(cls, python_type: Union[str, type, None, 'WorkflowNodeOutputType', 'PythonFunctionOutputArgumentType']) -> 'WorkflowNodeOutputType':
        if isinstance(python_type, WorkflowNodeOutputType):
            return python_type

        if isinstance(python_type, PythonFunctionOutputArgumentType):
            if python_type.value in cls.__members__:
                return cls(python_type.value)
            else:
                return cls.ANY

        if isinstance(python_type, type) or isinstance(python_type, str):
            python_type = python_type.__name__ if isinstance(python_type, type) else python_type
        else:
            python_type = type(python_type).__name__

        if python_type == 'int':
            return cls.INTEGER
        elif python_type == 'float':
            return cls.FLOAT
        elif python_type == 'str':
            return cls.STRING
        elif python_type == 'bool':
            return cls.BOOLEAN
        elif python_type == 'dict':
            return cls.DICT
        elif python_type == 'list':
            return cls.LIST
        elif python_type in ('NoneType', '_SpecialForm'):
            return cls.ANY
        raise ValueError(f'Unsupported output type: {python_type}')


class OcrMode(ApiEnum):
    AUTO = 'AUTO'
    DEFAULT = 'DEFAULT'
    LAYOUT = 'LAYOUT'
    SCANNED = 'SCANNED'
    COMPREHENSIVE = 'COMPREHENSIVE'
    COMPREHENSIVE_V2 = 'COMPREHENSIVE_V2'
    COMPREHENSIVE_TABLE_MD = 'COMPREHENSIVE_TABLE_MD'
    COMPREHENSIVE_FORM_MD = 'COMPREHENSIVE_FORM_MD'
    COMPREHENSIVE_FORM_AND_TABLE_MD = 'COMPREHENSIVE_FORM_AND_TABLE_MD'
    TESSERACT_FAST = 'TESSERACT_FAST'
    LLM = 'LLM'
    AUGMENTED_LLM = 'AUGMENTED_LLM'

    @classmethod
    def aws_ocr_modes(cls):
        return [cls.COMPREHENSIVE_V2, cls.COMPREHENSIVE_TABLE_MD, cls.COMPREHENSIVE_FORM_MD, cls.COMPREHENSIVE_FORM_AND_TABLE_MD]


class DocumentType(ApiEnum):
    SIMPLE_TEXT = 'SIMPLE_TEXT'                         # digital text
    TEXT = 'TEXT'                                       # general text with OCR
    TABLES_AND_FORMS = 'TABLES_AND_FORMS'               # tables and forms with OCR
    EMBEDDED_IMAGES = 'EMBEDDED_IMAGES'                 # embedded images with OCR TODO: remove?
    SCANNED_TEXT = 'SCANNED_TEXT'                       # scanned text with OCR
    COMPREHENSIVE_MARKDOWN = 'COMPREHENSIVE_MARKDOWN'   # comprehensive text with Gemini OCR

    @classmethod
    def is_ocr_forced(cls, document_type: 'DocumentType'):
        return document_type in [cls.TEXT, cls.TABLES_AND_FORMS, cls.EMBEDDED_IMAGES, cls.SCANNED_TEXT]


class StdDevThresholdType(ApiEnum):
    ABSOLUTE = 'ABSOLUTE'
    PERCENTILE = 'PERCENTILE'
    STDDEV = 'STDDEV'


class DataType(ApiEnum):
    INTEGER = 'integer'
    FLOAT = 'float'
    STRING = 'string'
    DATE = 'date'
    DATETIME = 'datetime'
    BOOLEAN = 'boolean'
    LIST = 'list'
    STRUCT = 'struct'
    NULL = 'null'
    BINARY = 'binary'


class AgentInterface(ApiEnum):
    # Duplicated in reainternal.enums, both should be kept in sync
    DEFAULT = 'DEFAULT'
    CHAT = 'CHAT'
    MATRIX = 'MATRIX'
    AUTONOMOUS = 'AUTONOMOUS'


class WorkflowNodeTemplateType(ApiEnum):
    TRIGGER = 'trigger'
    DEFAULT = 'default'


class ProjectConfigType(ApiEnum):
    CONSTRAINTS = 'CONSTRAINTS'
    CHAT_FEEDBACK = 'CHAT_FEEDBACK'
    REVIEW_MODE = 'REVIEW_MODE'


class CPUSize(ApiEnum):
    SMALL = 'small'
    MEDIUM = 'medium'
    LARGE = 'large'


class MemorySize(ApiEnum):
    SMALL = 16
    MEDIUM = 32
    LARGE = 64
    XLARGE = 128

    @classmethod
    def from_value(cls, value):
        sorted_members = sorted(cls, key=lambda mem: mem.value)
        for member in sorted_members:
            if member.value >= value:
                return member
        return None


class ResponseSectionType(ApiEnum):
    AGENT_FLOW_BUTTON = 'agent_flow_button'
    ATTACHMENTS = 'attachments'
    BASE64_IMAGE = 'base64_image'
    CHART = 'chart'
    CODE = 'code'
    COLLAPSIBLE_COMPONENT = 'collapsible_component'
    IMAGE_URL = 'image_url'
    RUNTIME_SCHEMA = 'runtime_schema'
    LIST = 'list'
    TABLE = 'table'
    TEXT = 'text'


class CodeLanguage(ApiEnum):
    PYTHON = 'python'
    SQL = 'sql'


class DeploymentConversationType(ApiEnum):
    CHAT_LLM = 'CHATLLM'
    SIMPLE_AGENT = 'SIMPLE_AGENT'
    COMPLEX_AGENT = 'COMPLEX_AGENT'
    WORKFLOW_AGENT = 'WORKFLOW_AGENT'
    COPILOT = 'COPILOT'
    AGENT_CONTROLLER = 'AGENT_CONTROLLER'
    CODE_LLM = 'CODE_LLM'
    CODE_LLM_AGENT = 'CODE_LLM_AGENT'
    CHAT_LLM_TASK = 'CHAT_LLM_TASK'
    COMPUTER_AGENT = 'COMPUTER_AGENT'
    SEARCH_LLM = 'SEARCH_LLM'
    APP_LLM = 'APP_LLM'
    TEST_AGENT = 'TEST_AGENT'
    SUPER_AGENT = 'SUPER_AGENT'


class AgentClientType(ApiEnum):
    CHAT_UI = 'CHAT_UI'
    MESSAGING_APP = 'MESSAGING_APP'
    API = 'API'
