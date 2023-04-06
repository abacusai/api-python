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
    USER_ITEM_SCORING = 'affinity'
    ANOMALY_DETECTION = 'anomaly_new'
    ANOMALY_OUTLIERS = 'anomaly'
    CLUSTERING = 'clustering'
    CLUSTERING_TIMESERIES = 'clustering_timeseries'
    CUMULATIVE_FORECASTING = 'cumulative_forecasting'
    NAMED_ENTITY_EXTRACTION = 'nlp_ner'
    NATURAL_LANGUAGE_SEARCH = 'nlp_search'
    SENTENCE_BOUNDARY_DETECTION = 'nlp_sentence_boundary_detection'
    SENTIMENT_DETECTION = 'nlp_sentiment'
    DOCUMENT_CLASSIFICATION = 'nlp_classification'
    DOCUMENT_SUMMARIZATION = 'nlp_summarization'
    DOCUMENT_VISUALIZATION = 'nlp_document_visualization'
    PERSONALIZATION = 'personalization'
    PREDICTIVE_MODELING = 'regression'
    FORECASTING = 'forecasting'
    CUSTOM_TRAINED_MODEL = 'plug_and_play'
    FEATURE_STORE = 'feature_store'
    IMAGE_CLASSIFICATION = 'vision_classification'
    OBJECT_DETECTION = 'vision_object_detection'
    IMAGE_VALUE_PREDICTION = 'vision_regression'
    MODEL_MONITORING = 'model_monitoring'
    LANGUAGE_DETECTION = 'language_detection'
    OPTIMIZATION = 'optimization'
    PRETRAINED_MODELS = 'pretrained'
    THEME_ANALYSIS = 'theme_analysis'


class SamplingMethodType(ApiEnum):
    N_SAMPLING = 'N_SAMPLING'
    PERCENT_SAMPLING = 'PERCENT_SAMPLING'


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
