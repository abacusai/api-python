import dataclasses

from . import enums
from .abstract import ApiClass, _ApiClassFactory


@dataclasses.dataclass
class BatchPredictionArgs(ApiClass):
    _support_kwargs: bool = dataclasses.field(default=True, repr=False, init=False)

    kwargs: dict = dataclasses.field(default_factory=dict)
    problem_type: enums.ProblemType = dataclasses.field(default=None)


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
class _BatchPredictionArgsFactory(_ApiClassFactory):
    config_abstract_class = BatchPredictionArgs
    config_class_key = 'problemType'
    config_class_map = {
        enums.ProblemType.PREDICTIVE_MODELING: PredictiveModelingBatchPredictionArgs,
    }
