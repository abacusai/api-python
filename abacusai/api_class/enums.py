from enum import Enum


class ApiEnum(Enum):
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
    ANOMALY_DETECTION = 'anomaly_new'
    ANOMALY_OUTLIERS = 'anomaly'
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
    RAPID = 'rapid',
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


class ExplainerType(Enum):
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


# Named Entity Recognition
class NERObjective(ApiEnum):
    LOG_LOSS = 'log_loss'
    AUC = 'auc'
    PRECISION = 'precision'
    RECALL = 'recall'
    ANNOTATIONS_PRECISION = 'annotations_precision'
    ANNOTATIONS_RECALL = 'annotations_recall'


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


class VectorStoreTextEncoder(ApiEnum):
    OPENAI = 'OPENAI'
    E5 = 'E5'
    SENTENCE_BERT = 'SENTENCE_BERT'
