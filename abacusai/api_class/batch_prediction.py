import dataclasses

from . import enums
from .abstract import ApiClass, _ApiClassFactory


@dataclasses.dataclass
class BatchPredictionArgs(ApiClass):
    _support_kwargs: bool = dataclasses.field(default=True, repr=False, init=False)

    kwargs: dict = dataclasses.field(default_factory=dict)
    problem_type: enums.ProblemType = dataclasses.field(default=None)

    @classmethod
    def _get_builder(cls):
        return _BatchPredictionArgsFactory


@dataclasses.dataclass
class AnomalyDetectionBatchPredictionArgs(BatchPredictionArgs):
    """
    Batch Prediction Config for the ANOMALY_DETECTION problem type
    Args:
        for_eval (bool): If True, the test fold which was created during training and used for metrics calculation will be used as input data. These predictions are hence, used for model evaluation.
        prediction_time_endpoint (str): The end point for predictions.
        prediction_time_range (int): Over what period of time should we make predictions (in seconds).
        minimum_anomaly_score (int): Exclude results with an anomaly score (1 in x event) below this threshold. Range: [1, 1_000_000_000_000].
        summary_mode (bool): Only show top anomalies per ID.
        attach_raw_data (bool): Return raw data along with anomalies.
        small_batch (bool): Size of batch data guaranteed to be small.
    """
    for_eval: bool = dataclasses.field(default=None)
    prediction_time_endpoint: str = dataclasses.field(default=None)
    prediction_time_range: int = dataclasses.field(default=None)
    minimum_anomaly_score: int = dataclasses.field(default=None)
    summary_mode: bool = dataclasses.field(default=None)
    attach_raw_data: bool = dataclasses.field(default=None)
    small_batch: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.ANOMALY_DETECTION


@dataclasses.dataclass
class AnomalyOutliersBatchPredictionArgs(BatchPredictionArgs):
    """
    Batch Prediction Config for the ANOMALY_OUTLIERS problem type
    Args:
        for_eval (bool): If True, the test fold which was created during training and used for metrics calculation will be used as input data. These predictions are hence, used for model evaluation.
        threshold (float): The threshold for detecting an anomaly. Range: [0.8, 0.99]
    """
    for_eval: bool = dataclasses.field(default=None)
    threshold: float = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.ANOMALY_OUTLIERS


@dataclasses.dataclass
class ForecastingBatchPredictionArgs(BatchPredictionArgs):
    """
    Batch Prediction Config for the FORECASTING problem type
    Args:
       for_eval (bool): If True, the test fold which was created during training and used for metrics calculation will be used as input data. These predictions are hence, used for model evaluation
       predictions_start_date (str): The start date for predictions.
       use_prediction_offset (bool): If True, use prediction offset.
       start_date_offset (int): Sets prediction start date as this offset relative to the prediction start date.
       forecasting_horizon (int): The number of timestamps to predict in the future. Range: [1, 1000].
       item_attributes_to_include_in_the_result (list): List of columns to include in the prediction output.
       explain_predictions (bool): If True, explain predictions for the forecast.
    """
    for_eval: bool = dataclasses.field(default=None)
    predictions_start_date: str = dataclasses.field(default=None)
    use_prediction_offset: bool = dataclasses.field(default=None)
    start_date_offset: int = dataclasses.field(default=None)
    forecasting_horizon: int = dataclasses.field(default=None)
    item_attributes_to_include_in_the_result: list = dataclasses.field(default=None)
    explain_predictions: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.FORECASTING


@dataclasses.dataclass
class NamedEntityExtractionBatchPredictionArgs(BatchPredictionArgs):
    """
    Batch Prediction Config for the NAMED_ENTITY_EXTRACTION problem type
    Args:
       for_eval (bool): If True, the test fold which was created during training and used for metrics calculation will be used as input data. These predictions are hence, used for model evaluation.
       verbose_predictions (bool): Return prediction inputs, predicted annotations and token label probabilities.
    """
    for_eval: bool = dataclasses.field(default=None)
    verbose_predictions: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.NAMED_ENTITY_EXTRACTION


@dataclasses.dataclass
class PersonalizationBatchPredictionArgs(BatchPredictionArgs):
    """
    Batch Prediction Config for the PERSONALIZATION problem type
    Args:
        for_eval (bool): If True, the test fold which was created during training and used for metrics calculation will be used as input data. These predictions are hence, used for model evaluation.
        number_of_items (int): Number of items to recommend.
        result_columns (list): List of columns to include in the prediction output.
        score_field (str): If specified, relative item scores will be returned using a field with this name
    """
    for_eval: bool = dataclasses.field(default=None)
    number_of_items: int = dataclasses.field(default=None, metadata={'alias': 'num_items'})
    item_attributes_to_include_in_the_result: list = dataclasses.field(default=None, metadata={'alias': 'result_columns'})
    score_field: str = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.PERSONALIZATION


@dataclasses.dataclass
class PredictiveModelingBatchPredictionArgs(BatchPredictionArgs):
    """
    Batch Prediction Config for the PREDICTIVE_MODELING problem type
    Args:
       for_eval (bool): If True, the test fold which was created during training and used for metrics calculation will be used as input data. These predictions are hence, used for model evaluation.
       explainer_type (enums.ExplainerType): The type of explainer to use to generate explanations on the batch prediction.
       number_of_samples_to_use_for_explainer (int): Number Of Samples To Use For Kernel Explainer.
       include_multi_class_explanations (bool): If True, Includes explanations for all classes in multi-class classification.
       features_considered_constant_for_explanations (str): Comma separate list of fields to treat as constant in SHAP explanations.
       importance_of_records_in_nested_columns (str): Returns importance of each index in the specified nested column instead of SHAP column explanations.
       explanation_filter_lower_bound (float): If set explanations will be limited to predictions above this value, Range: [0, 1].
       explanation_filter_upper_bound (float): If set explanations will be limited to predictions below this value, Range: [0, 1].
       bound_label (str): For classification problems specifies the label to which the explanation bounds are applied.
       output_columns (list): A list of column names to include in the prediction result.
    """
    for_eval: bool = dataclasses.field(default=None)
    explainer_type: enums.ExplainerType = dataclasses.field(default=None)
    number_of_samples_to_use_for_explainer: int = dataclasses.field(default=None)
    include_multi_class_explanations: bool = dataclasses.field(default=None)
    features_considered_constant_for_explanations: str = dataclasses.field(default=None)
    importance_of_records_in_nested_columns: str = dataclasses.field(default=None)
    explanation_filter_lower_bound: float = dataclasses.field(default=None)
    explanation_filter_upper_bound: float = dataclasses.field(default=None)
    explanation_filter_label: str = dataclasses.field(default=None)
    output_columns: list = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.PREDICTIVE_MODELING


@dataclasses.dataclass
class PretrainedModelsBatchPredictionArgs(BatchPredictionArgs):
    """
    Batch Prediction Config for the PRETRAINED_MODELS problem type
    Args:
        for_eval (bool): If True, the test fold which was created during training and used for metrics calculation will be used as input data. These predictions are hence, used for model evaluation.
        files_output_location_prefix (str): The output location prefix for the files.
        channel_id_to_label_map (str): JSON string for the map from channel ids to their labels.
    """
    for_eval: bool = dataclasses.field(default=None)
    files_output_location_prefix: str = dataclasses.field(default=None)
    channel_id_to_label_map: str = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.PRETRAINED_MODELS


@dataclasses.dataclass
class SentenceBoundaryDetectionBatchPredictionArgs(BatchPredictionArgs):
    """
    Batch Prediction Config for the SENTENCE_BOUNDARY_DETECTION problem type
    Args:
       for_eval (bool): If True, the test fold which was created during training and used for metrics calculation will be used as input data. These predictions are hence, used for model evaluation
       explode_output (bool): Explode data so there is one sentence per row.
    """
    for_eval: bool = dataclasses.field(default=None)
    explode_output: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.SENTENCE_BOUNDARY_DETECTION


@dataclasses.dataclass
class ThemeAnalysisBatchPredictionArgs(BatchPredictionArgs):
    """
    Batch Prediction Config for the THEME_ANALYSIS problem type
    Args:
        for_eval (bool): If True, the test fold which was created during training and used for metrics calculation will be used as input data. These predictions are hence, used for model evaluation.
        analysis_frequency (str): The length of each analysis interval.
        start_date (str): The end point for predictions.
        analysis_days (int): How many days to analyze.
    """
    for_eval: bool = dataclasses.field(default=None)
    analysis_frequency: str = dataclasses.field(default=None)
    start_date: str = dataclasses.field(default=None)
    analysis_days: int = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.THEME_ANALYSIS


@dataclasses.dataclass
class ChatLLMBatchPredictionArgs(BatchPredictionArgs):
    """
    Batch Prediction Config for the ChatLLM problem type
    Args:
        for_eval (bool): If True, the test fold which was created during training and used for metrics calculation will be used as input data. These predictions are hence, used for model evaluation.
    """
    for_eval: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.CHAT_LLM


@dataclasses.dataclass
class _BatchPredictionArgsFactory(_ApiClassFactory):
    config_abstract_class = BatchPredictionArgs
    config_class_key = 'problemType'
    config_class_map = {
        enums.ProblemType.ANOMALY_DETECTION: AnomalyDetectionBatchPredictionArgs,
        enums.ProblemType.ANOMALY_OUTLIERS: AnomalyOutliersBatchPredictionArgs,
        enums.ProblemType.FORECASTING: ForecastingBatchPredictionArgs,
        enums.ProblemType.NAMED_ENTITY_EXTRACTION: NamedEntityExtractionBatchPredictionArgs,
        enums.ProblemType.PERSONALIZATION: PersonalizationBatchPredictionArgs,
        enums.ProblemType.PREDICTIVE_MODELING: PredictiveModelingBatchPredictionArgs,
        enums.ProblemType.PRETRAINED_MODELS: PretrainedModelsBatchPredictionArgs,
        enums.ProblemType.SENTENCE_BOUNDARY_DETECTION: SentenceBoundaryDetectionBatchPredictionArgs,
        enums.ProblemType.THEME_ANALYSIS: ThemeAnalysisBatchPredictionArgs,
        enums.ProblemType.CHAT_LLM: ChatLLMBatchPredictionArgs,
    }
