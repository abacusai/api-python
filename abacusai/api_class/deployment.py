import dataclasses

from . import enums
from .abstract import ApiClass, _ApiClassFactory


@dataclasses.dataclass
class PredictionArguments(ApiClass):
    """
    An abstract class for prediction arguments specific to problem type.
    """
    _support_kwargs: bool = dataclasses.field(default=True, repr=False, init=False)

    kwargs: dict = dataclasses.field(default_factory=dict)
    problem_type: enums.ProblemType = dataclasses.field(default=None, repr=False, init=False)

    @classmethod
    def _get_builder(cls):
        return _PredictionArgumentsFactory


@dataclasses.dataclass
class OptimizationPredictionArguments(PredictionArguments):
    """
    Prediction arguments for the OPTIMIZATION problem type

    Args:
        forced_assignments (dict): Set of assignments to force and resolve before returning query results.
        solve_time_limit_seconds (float): Maximum time in seconds to spend solving the query.
        include_all_assignments (bool): If True, will return all assignments, including assignments with value 0. Default is False.
    """
    forced_assignments: dict = dataclasses.field(default=None)
    solve_time_limit_seconds: float = dataclasses.field(default=None)
    include_all_assignments: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.OPTIMIZATION


@dataclasses.dataclass
class TimeseriesAnomalyPredictionArguments(PredictionArguments):
    """
    Prediction arguments for the TS_ANOMALY problem type

    Args:
        start_timestamp (str): Timestamp from which anomalies have to be detected in the training data
        end_timestamp (str): Timestamp to which anomalies have to be detected in the training data
        get_all_item_data (bool): If True, anomaly detection has to be performed on all the data related to input ids
    """
    start_timestamp: str = dataclasses.field(default=None)
    end_timestamp: str = dataclasses.field(default=None)
    get_all_item_data: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.TS_ANOMALY


@dataclasses.dataclass
class ChatLLMPredictionArguments(PredictionArguments):
    """
    Prediction arguments for the CHAT_LLM problem type

    Args:
        llm_name (str): Name of the specific LLM backend to use to power the chat experience.
        num_completion_tokens (int): Default for maximum number of tokens for chat answers.
        system_message (str): The generative LLM system message.
        temperature (float): The generative LLM temperature.
        search_score_cutoff (float): Cutoff for the document retriever score. Matching search results below this score will be ignored.
        ignore_documents (bool): If True, will ignore any documents and search results, and only use the messages to generate a response.
    """
    llm_name: str = dataclasses.field(default=None)
    num_completion_tokens: int = dataclasses.field(default=None)
    system_message: str = dataclasses.field(default=None)
    temperature: float = dataclasses.field(default=None)
    search_score_cutoff: float = dataclasses.field(default=None)
    ignore_documents: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.CHAT_LLM


@dataclasses.dataclass
class RegressionPredictionArguments(PredictionArguments):
    """
    Prediction arguments for the PREDICTIVE_MODELING problem type

    Args:
        explain_predictions (bool): If true, will explain predictions.
        explainer_type (str): Type of explainer to use for explanations.
    """
    explain_predictions: bool = dataclasses.field(default=None)
    explainer_type: str = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.PREDICTIVE_MODELING


@dataclasses.dataclass
class ForecastingPredictionArguments(PredictionArguments):
    """
    Prediction arguments for the FORECASTING problem type

    Args:
        num_predictions (int): The number of timestamps to predict in the future.
        prediction_start (str): The start date for predictions (e.g., "2015-08-01T00:00:00" as input for mid-night of 2015-08-01).
        explain_predictions (bool): If True, explain predictions for forecasting.
        explainer_type (str): Type of explainer to use for explanations.
        get_item_data (bool): If True, will return the data corresponding to items as well.
    """
    num_predictions: int = dataclasses.field(default=None)
    prediction_start: str = dataclasses.field(default=None)
    explain_predictions: bool = dataclasses.field(default=None)
    explainer_type: str = dataclasses.field(default=None)
    get_item_data: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.FORECASTING


@dataclasses.dataclass
class CumulativeForecastingPredictionArguments(PredictionArguments):
    """
    Prediction arguments for the CUMULATIVE_FORECASTING problem type

    Args:
        num_predictions (int): The number of timestamps to predict in the future.
        prediction_start (str): The start date for predictions (e.g., "2015-08-01T00:00:00" as input for mid-night of 2015-08-01).
        explain_predictions (bool): If True, explain predictions for forecasting.
        explainer_type (str): Type of explainer to use for explanations.
        get_item_data (bool): If True, will return the data corresponding to items as well.
    """
    num_predictions: int = dataclasses.field(default=None)
    prediction_start: str = dataclasses.field(default=None)
    explain_predictions: bool = dataclasses.field(default=None)
    explainer_type: str = dataclasses.field(default=None)
    get_item_data: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.CUMULATIVE_FORECASTING


@dataclasses.dataclass
class NaturalLanguageSearchPredictionArguments(PredictionArguments):
    """
    Prediction arguments for the NATURAL_LANGUAGE_SEARCH problem type

    Args:
        llm_name (str): Name of the specific LLM backend to use to power the chat experience.
        num_completion_tokens (int): Default for maximum number of tokens for chat answers.
        system_message (str): The generative LLM system message.
        temperature (float): The generative LLM temperature.
        search_score_cutoff (float): Cutoff for the document retriever score. Matching search results below this score will be ignored.
        ignore_documents (bool): If True, will ignore any documents and search results, and only use the messages to generate a response.
    """
    llm_name: str = dataclasses.field(default=None)
    num_completion_tokens: int = dataclasses.field(default=None)
    system_message: str = dataclasses.field(default=None)
    temperature: float = dataclasses.field(default=None)
    search_score_cutoff: float = dataclasses.field(default=None)
    ignore_documents: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.NATURAL_LANGUAGE_SEARCH


@dataclasses.dataclass
class FeatureStorePredictionArguments(PredictionArguments):
    """
    Prediction arguments for the FEATURE_STORE problem type

    Args:
        limit_results (int): If provided, will limit the number of results to the value specified.
    """
    limit_results: int = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.FEATURE_STORE


@dataclasses.dataclass
class _PredictionArgumentsFactory(_ApiClassFactory):
    config_abstract_class = PredictionArguments
    config_class_key = 'problem_type'
    config_class_map = {
        enums.ProblemType.CHAT_LLM: ChatLLMPredictionArguments,
        enums.ProblemType.CUMULATIVE_FORECASTING: CumulativeForecastingPredictionArguments,
        enums.ProblemType.FORECASTING: ForecastingPredictionArguments,
        enums.ProblemType.FEATURE_STORE: FeatureStorePredictionArguments,
        enums.ProblemType.NATURAL_LANGUAGE_SEARCH: NaturalLanguageSearchPredictionArguments,
        enums.ProblemType.OPTIMIZATION: OptimizationPredictionArguments,
        enums.ProblemType.PREDICTIVE_MODELING: RegressionPredictionArguments,
        enums.ProblemType.TS_ANOMALY: TimeseriesAnomalyPredictionArguments,
    }
