import dataclasses

from . import enums
from .abstract import ApiClass, _ApiClassFactory


@dataclasses.dataclass
class BatchPredictionArgs(ApiClass):
    """
    An abstract class for Batch Prediction args specific to problem type.
    """
    _support_kwargs: bool = dataclasses.field(default=True, repr=False, init=False)

    kwargs: dict = dataclasses.field(default_factory=dict)
    problem_type: enums.ProblemType = dataclasses.field(default=None, repr=False, init=False)

    @classmethod
    def _get_builder(cls):
        return _BatchPredictionArgsFactory


@dataclasses.dataclass
class ForecastingBatchPredictionArgs(BatchPredictionArgs):
    """
    Batch Prediction Config for the FORECASTING problem type

    Args:
       for_eval (bool): If True, the test fold which was created during training and used for metrics calculation will be used as input data. These predictions are hence, used for model evaluation
       predictions_start_date (str): The start date for predictions. Accepts timestamp integers and strings in many standard formats such as YYYY-MM-DD, YYYY-MM-DD HH:MM:SS, or YYYY-MM-DDTHH:MM:SS. If not specified, the prediction start date will be automatically defined.
       use_prediction_offset (bool): If True, use prediction offset.
       start_date_offset (int): Sets prediction start date as this offset relative to the prediction start date.
       forecasting_horizon (int): The number of timestamps to predict in the future. Range: [1, 1000].
       item_attributes_to_include_in_the_result (list): List of columns to include in the prediction output.
       explain_predictions (bool): If True, calculates explanations for the forecasted values along with predictions.
       create_monitor (bool): Controls whether to automatically create a monitor to calculate the drift each time the batch prediction is run. Defaults to true if not specified.
    """
    for_eval: bool = dataclasses.field(default=None)
    predictions_start_date: str = dataclasses.field(default=None)
    use_prediction_offset: bool = dataclasses.field(default=None)
    start_date_offset: int = dataclasses.field(default=None)
    forecasting_horizon: int = dataclasses.field(default=None)
    item_attributes_to_include_in_the_result: list = dataclasses.field(default=None)
    explain_predictions: bool = dataclasses.field(default=None)
    create_monitor: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.FORECASTING


@dataclasses.dataclass
class NamedEntityExtractionBatchPredictionArgs(BatchPredictionArgs):
    """
    Batch Prediction Config for the NAMED_ENTITY_EXTRACTION problem type

    Args:
       for_eval (bool): If True, the test fold which was created during training and used for metrics calculation will be used as input data. These predictions are hence, used for model evaluation.
    """
    for_eval: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.NAMED_ENTITY_EXTRACTION


@dataclasses.dataclass
class PersonalizationBatchPredictionArgs(BatchPredictionArgs):
    """
    Batch Prediction Config for the PERSONALIZATION problem type

    Args:
        for_eval (bool): If True, the test fold which was created during training and used for metrics calculation will be used as input data. These predictions are hence, used for model evaluation.
        number_of_items (int): Number of items to recommend.
        item_attributes_to_include_in_the_result (list): List of columns to include in the prediction output.
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
       explanation_filter_label (str): For classification problems specifies the label to which the explanation bounds are applied.
       output_columns (list): A list of column names to include in the prediction result.
       explain_predictions (bool): If True, calculates explanations for the predicted values along with predictions.
       create_monitor (bool): Controls whether to automatically create a monitor to calculate the drift each time the batch prediction is run. Defaults to true if not specified.
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
    explain_predictions: bool = dataclasses.field(default=None)
    create_monitor: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.PREDICTIVE_MODELING


@dataclasses.dataclass
class PretrainedModelsBatchPredictionArgs(BatchPredictionArgs):
    """
    Batch Prediction Config for the PRETRAINED_MODELS problem type

    Args:
        for_eval (bool): If True, the test fold which was created during training and used for metrics calculation will be used as input data. These predictions are hence, used for model evaluation.
        files_input_location (str): The input location for the files.
        files_output_location_prefix (str): The output location prefix for the files.
        channel_id_to_label_map (str): JSON string for the map from channel ids to their labels.
    """
    for_eval: bool = dataclasses.field(default=None)
    files_input_location: str = dataclasses.field(default=None)
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
class TrainablePlugAndPlayBatchPredictionArgs(BatchPredictionArgs):
    """
    Batch Prediction Config for the TrainablePlugAndPlay problem type

    Args:
        for_eval (bool): If True, the test fold which was created during training and used for metrics calculation will be used as input data. These predictions are hence, used for model evaluation.
        create_monitor (bool): Controls whether to automatically create a monitor to calculate the drift each time the batch prediction is run. Defaults to true if not specified.
    """
    for_eval: bool = dataclasses.field(default=None)
    create_monitor: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.CUSTOM_ALGORITHM


@dataclasses.dataclass
class AIAgentBatchPredictionArgs(BatchPredictionArgs):
    """
    Batch Prediction Config for the AIAgents problem type
    """

    def __post_init__(self):
        self.problem_type = enums.ProblemType.AI_AGENT


@dataclasses.dataclass
class _BatchPredictionArgsFactory(_ApiClassFactory):
    config_abstract_class = BatchPredictionArgs
    config_class_key = 'problem_type'
    config_class_map = {
        enums.ProblemType.FORECASTING: ForecastingBatchPredictionArgs,
        enums.ProblemType.NAMED_ENTITY_EXTRACTION: NamedEntityExtractionBatchPredictionArgs,
        enums.ProblemType.PERSONALIZATION: PersonalizationBatchPredictionArgs,
        enums.ProblemType.PREDICTIVE_MODELING: PredictiveModelingBatchPredictionArgs,
        enums.ProblemType.PRETRAINED_MODELS: PretrainedModelsBatchPredictionArgs,
        enums.ProblemType.SENTENCE_BOUNDARY_DETECTION: SentenceBoundaryDetectionBatchPredictionArgs,
        enums.ProblemType.THEME_ANALYSIS: ThemeAnalysisBatchPredictionArgs,
        enums.ProblemType.CHAT_LLM: ChatLLMBatchPredictionArgs,
        enums.ProblemType.CUSTOM_ALGORITHM: TrainablePlugAndPlayBatchPredictionArgs,
        enums.ProblemType.AI_AGENT: AIAgentBatchPredictionArgs,
    }
