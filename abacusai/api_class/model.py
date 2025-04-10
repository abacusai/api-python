import dataclasses
from typing import Dict, List

from . import enums
from .abstract import ApiClass, _ApiClassFactory


@dataclasses.dataclass
class TrainingConfig(ApiClass):
    """
    An abstract class for the training config options used to train the model.
    """
    _upper_snake_case_keys: bool = dataclasses.field(default=True, repr=False, init=False)
    _support_kwargs: bool = dataclasses.field(default=True, repr=False, init=False)

    kwargs: dict = dataclasses.field(default_factory=dict)
    problem_type: enums.ProblemType = dataclasses.field(default=None, repr=False, init=False)
    algorithm: str = dataclasses.field(default=None)

    @classmethod
    def _get_builder(cls):
        return _TrainingConfigFactory


@dataclasses.dataclass
class PersonalizationTrainingConfig(TrainingConfig):
    """
    Training config for the PERSONALIZATION problem type

    Args:
        objective (PersonalizationObjective): Ranking scheme used to select final best model.
        sort_objective (PersonalizationObjective): Ranking scheme used to sort models on the metrics page.
        training_mode (PersonalizationTrainingMode): whether to train in production or experimental mode. Defaults to EXP.
        target_action_types (List[str]): List of action types to use as targets for training.
        target_action_weights (Dict[str, float]): Dictionary of action types to weights for training.
        session_event_types (List[str]): List of event types to treat as occurrences of sessions.
        test_split (int): Percent of dataset to use for test data. We support using a range between 6% to 20% of your dataset to use as test data.
        recent_days_for_training (int): Limit training data to a certain latest number of days.
        training_start_date (str): Only consider training interaction data after this date. Specified in the timezone of the dataset.
        test_on_user_split (bool): Use user splits instead of using time splits, when validating and testing the model.
        test_split_on_last_k_items (bool): Use last k items instead of global timestamp splits, when validating and testing the model.
        test_last_items_length (int): Number of items to leave out for each user when using leave k out folds.
        test_window_length_hours (int): Duration (in hours) of most recent time window to use when validating and testing the model.
        explicit_time_split (bool): Sets an explicit time-based test boundary.
        test_row_indicator (str): Column indicating which rows to use for training (TRAIN), validation (VAL) and testing (TEST).
        full_data_retraining (bool): Train models separately with all the data.
        sequential_training (bool): Train a mode sequentially through time.
        data_split_feature_group_table_name (str): Specify the table name of the feature group to export training data with the fold column.
        optimized_event_type (str): The final event type to optimize for and compute metrics on.
        dropout_rate (int): Dropout rate for neural network.
        batch_size (BatchSize): Batch size for neural network.
        disable_transformer (bool): Disable training the transformer algorithm.
        disable_gpu (boo): Disable training on GPU.
        filter_history (bool): Do not recommend items the user has already interacted with.
        action_types_exclusion_days (Dict[str, float]): Mapping from action type to number of days for which we exclude previously interacted items from prediction
        session_dedupe_mins (float): Minimum number of minutes between two sessions for a user.
        max_history_length (int): Maximum length of user-item history to include user in training examples.
        compute_rerank_metrics (bool): Compute metrics based on rerank results.
        add_time_features (bool): Include interaction time as a feature.
        disable_timestamp_scalar_features (bool): Exclude timestamp scalar features.
        compute_session_metrics (bool): Evaluate models based on how well they are able to predict the next session of interactions.
        max_user_history_len_percentile (int): Filter out users with history length above this percentile.
        downsample_item_popularity_percentile (float): Downsample items more popular than this percentile.
        use_user_id_feature (bool): Use user id as a feature in CTR models.
        min_item_history (int): Minimum number of interactions an item must have to be included in training.
        query_column (str): Name of column in the interactions table that represents a natural language query, e.g. 'blue t-shirt'.
        item_query_column (str): Name of column in the item catalog that will be matched to the query column in the interactions table.
        include_item_id_feature (bool): Add Item-Id to the input features of the model. Applicable for Embedding distance and CTR models.
    """
    # top-level params
    objective: enums.PersonalizationObjective = dataclasses.field(default=None)
    sort_objective: enums.PersonalizationObjective = dataclasses.field(default=None)
    training_mode: enums.PersonalizationTrainingMode = dataclasses.field(default=None)

    # advanced options
    # interactions
    target_action_types: List[str] = dataclasses.field(default=None)
    target_action_weights: Dict[str, float] = dataclasses.field(default=None)
    session_event_types: List[str] = dataclasses.field(default=None)

    # data split
    test_split: int = dataclasses.field(default=None)
    recent_days_for_training: int = dataclasses.field(default=None)
    training_start_date: str = dataclasses.field(default=None)
    test_on_user_split: bool = dataclasses.field(default=None)
    test_split_on_last_k_items: bool = dataclasses.field(default=None)
    test_last_items_length: int = dataclasses.field(default=None)
    test_window_length_hours: int = dataclasses.field(default=None)
    explicit_time_split: bool = dataclasses.field(default=None)
    test_row_indicator: str = dataclasses.field(default=None)
    full_data_retraining: bool = dataclasses.field(default=None)
    sequential_training: bool = dataclasses.field(default=None)
    data_split_feature_group_table_name: str = dataclasses.field(default=None)
    optimized_event_type: str = dataclasses.field(default=None)

    # neural network
    dropout_rate: int = dataclasses.field(default=None)
    batch_size: enums.BatchSize = dataclasses.field(default=None)
    disable_transformer: bool = dataclasses.field(default=None)
    disable_gpu: bool = dataclasses.field(default=None)

    # prediction
    filter_history: bool = dataclasses.field(default=None)
    action_types_exclusion_days: Dict[str, float] = dataclasses.field(default=None)

    # data distribution
    max_history_length: int = dataclasses.field(default=None)
    compute_rerank_metrics: bool = dataclasses.field(default=None)
    add_time_features: bool = dataclasses.field(default=None)
    disable_timestamp_scalar_features: bool = dataclasses.field(default=None)
    compute_session_metrics: bool = dataclasses.field(default=None)
    query_column: str = dataclasses.field(default=None)
    item_query_column: str = dataclasses.field(default=None)
    use_user_id_feature: bool = dataclasses.field(default=None)
    session_dedupe_mins: float = dataclasses.field(default=None)
    include_item_id_feature: bool = dataclasses.field(default=None)

    # outliers
    max_user_history_len_percentile: int = dataclasses.field(default=None)
    downsample_item_popularity_percentile: float = dataclasses.field(default=None)
    min_item_history: int = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.PERSONALIZATION


@dataclasses.dataclass
class RegressionTrainingConfig(TrainingConfig):
    """
    Training config for the PREDICTIVE_MODELING problem type

    Args:
        objective (RegressionObjective): Ranking scheme used to select final best model.
        sort_objective (RegressionObjective): Ranking scheme used to sort models on the metrics page.
        tree_hpo_mode: (RegressionTreeHPOMode): Turning off Rapid Experimentation will take longer to train.
        type_of_split (RegressionTypeOfSplit): Type of data splitting into train/test (validation also).
        test_split (int): Percent of dataset to use for test data. We support using a range between 5% to 20% of your dataset to use as test data.
        disable_test_val_fold (bool): Do not create a TEST_VAL set. All records which would be part of the TEST_VAL fold otherwise, remain in the TEST fold.
        k_fold_cross_validation (bool): Use this to force k-fold cross validation bagging on or off.
        num_cv_folds (int): Specify the value of k in k-fold cross validation.
        timestamp_based_splitting_column (str): Timestamp column selected for splitting into test and train.
        timestamp_based_splitting_method (RegressionTimeSplitMethod): Method of selecting TEST set, top percentile wise or after a given timestamp.
        test_splitting_timestamp (str): Rows with timestamp greater than this will be considered to be in the test set.
        sampling_unit_keys (List[str]): Constrain train/test separation to partition a column.
        test_row_indicator (str): Column indicating which rows to use for training (TRAIN) and testing (TEST). Validation (VAL) can also be specified.
        full_data_retraining (bool): Train models separately with all the data.
        rebalance_classes (bool): Class weights are computed as the inverse of the class frequency from the training dataset when this option is selected as "Yes". It is useful when the classes in the dataset are unbalanced.
                                  Re-balancing classes generally boosts recall at the cost of precision on rare classes.
        rare_class_augmentation_threshold (float): Augments any rare class whose relative frequency with respect to the most frequent class is less than this threshold. Default = 0.1 for classification problems with rare classes.
        augmentation_strategy (RegressionAugmentationStrategy): Strategy to deal with class imbalance and data augmentation.
        training_rows_downsample_ratio (float): Uses this ratio to train on a sample of the dataset provided.
        active_labels_column (str): Specify a column to use as the active columns in a multi label setting.
        min_categorical_count (int): Minimum threshold to consider a value different from the unknown placeholder.
        sample_weight (str): Specify a column to use as the weight of a sample for training and eval.
        numeric_clipping_percentile (float): Uses this option to clip the top and bottom x percentile of numeric feature columns where x is the value of this option.
        target_transform (RegressionTargetTransform): Specify a transform (e.g. log, quantile) to apply to the target variable.
        ignore_datetime_features (bool): Remove all datetime features from the model. Useful while generalizing to different time periods.
        max_text_words (int): Maximum number of words to use from text fields.
        perform_feature_selection (bool): If enabled, additional algorithms which support feature selection as a pretraining step will be trained separately with the selected subset of features. The details about their selected features can be found in their respective logs.
        feature_selection_intensity (int): This determines the strictness with which features will be filtered out. 1 being very lenient (more features kept), 100 being very strict.
        batch_size (BatchSize): Batch size.
        dropout_rate (int): Dropout percentage rate.
        pretrained_model_name (str): Enable algorithms which process text using pretrained multilingual NLP models.
        pretrained_llm_name (str): Enable algorithms which process text using pretrained large language models.
        is_multilingual (bool): Enable algorithms which process text using pretrained multilingual NLP models.
        loss_function (RegressionLossFunction): Loss function to be used as objective for model training.
        loss_parameters (str): Loss function params in format <key>=<value>;<key>=<value>;.....
        target_encode_categoricals (bool): Use this to turn target encoding on categorical features on or off.
        drop_original_categoricals (bool): This option helps us choose whether to also feed the original label encoded categorical columns to the mdoels along with their target encoded versions.
        monotonically_increasing_features (List[str]): Constrain the model such that it behaves as if the target feature is monotonically increasing with the selected features
        monotonically_decreasing_features (List[str]): Constrain the model such that it behaves as if the target feature is monotonically decreasing with the selected features
        data_split_feature_group_table_name (str): Specify the table name of the feature group to export training data with the fold column.
        custom_loss_functions (List[str]): Registered custom losses available for selection.
        custom_metrics (List[str]): Registered custom metrics available for selection.
        partial_dependence_analysis (PartialDependenceAnalysis): Specify whether to run partial dependence plots for all features or only some features.
        do_masked_language_model_pretraining (bool): Specify whether to run a masked language model unsupervised pretraining step before supervized training in certain supported algorithms which use BERT-like backbones.
        max_tokens_in_sentence (int): Specify the max tokens to be kept in a sentence based on the truncation strategy.
        truncation_strategy (str): What strategy to use to deal with text rows with more than a given number of tokens (if num of tokens is more than "max_tokens_in_sentence").
     """

    objective: enums.RegressionObjective = dataclasses.field(default=None)
    sort_objective: enums.RegressionObjective = dataclasses.field(default=None)
    tree_hpo_mode: enums.RegressionTreeHPOMode = dataclasses.field(default=None)
    partial_dependence_analysis: enums.PartialDependenceAnalysis = dataclasses.field(default=None)

    # data split related
    type_of_split: enums.RegressionTypeOfSplit = dataclasses.field(default=None)
    test_split: int = dataclasses.field(default=None)
    disable_test_val_fold: bool = dataclasses.field(default=None)
    k_fold_cross_validation: bool = dataclasses.field(default=None)
    num_cv_folds: int = dataclasses.field(default=None)
    timestamp_based_splitting_column: str = dataclasses.field(default=None)
    timestamp_based_splitting_method: enums.RegressionTimeSplitMethod = dataclasses.field(default=None)
    test_splitting_timestamp: str = dataclasses.field(default=None)
    sampling_unit_keys: List[str] = dataclasses.field(default=None)
    test_row_indicator: str = dataclasses.field(default=None)
    full_data_retraining: bool = dataclasses.field(default=None)

    # data augmentation
    rebalance_classes: bool = dataclasses.field(default=None)
    rare_class_augmentation_threshold: float = dataclasses.field(default=None)
    augmentation_strategy: enums.RegressionAugmentationStrategy = dataclasses.field(default=None)
    training_rows_downsample_ratio: float = dataclasses.field(default=None)

    # multivalue categorical
    active_labels_column: str = dataclasses.field(default=None)

    # features and columns
    min_categorical_count: int = dataclasses.field(default=None)
    sample_weight: str = dataclasses.field(default=None)
    numeric_clipping_percentile: float = dataclasses.field(default=None)
    target_transform: enums.RegressionTargetTransform = dataclasses.field(default=None)
    ignore_datetime_features: bool = dataclasses.field(default=None)
    max_text_words: int = dataclasses.field(default=None)
    perform_feature_selection: bool = dataclasses.field(default=None)
    feature_selection_intensity: int = dataclasses.field(default=None)

    # neural network
    batch_size: enums.BatchSize = dataclasses.field(default=None)
    dropout_rate: int = dataclasses.field(default=None)
    pretrained_model_name: str = dataclasses.field(default=None)
    pretrained_llm_name: str = dataclasses.field(default=None)
    is_multilingual: bool = dataclasses.field(default=None)
    do_masked_language_model_pretraining: bool = dataclasses.field(default=None)
    max_tokens_in_sentence: int = dataclasses.field(default=None)
    truncation_strategy: str = dataclasses.field(default=None)

    # loss function
    loss_function: enums.RegressionLossFunction = dataclasses.field(default=None)
    loss_parameters: str = dataclasses.field(default=None)

    # target encoding
    target_encode_categoricals: bool = dataclasses.field(default=None)
    drop_original_categoricals: bool = dataclasses.field(default=None)

    # monotonic features
    monotonically_increasing_features: List[str] = dataclasses.field(default=None)
    monotonically_decreasing_features: List[str] = dataclasses.field(default=None)

    # Others
    data_split_feature_group_table_name: str = dataclasses.field(default=None)
    custom_loss_functions: List[str] = dataclasses.field(default=None)
    custom_metrics: List[str] = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.PREDICTIVE_MODELING


@dataclasses.dataclass
class ForecastingTrainingConfig(TrainingConfig):
    """
    Training config for the FORECASTING problem type

    Args:
        prediction_length (int): How many timesteps in the future to predict.
        objective (ForecastingObjective): Ranking scheme used to select final best model.
        sort_objective (ForecastingObjective): Ranking scheme used to sort models on the metrics page.
        forecast_frequency (ForecastingFrequency): Forecast frequency.
        probability_quantiles (List[float]): Prediction quantiles.
        force_prediction_length (int): Force length of test window to be the same as prediction length.
        filter_items (bool): Filter items with small history and volume.
        enable_feature_selection (bool): Enable feature selection.
        enable_padding (bool): Pad series to the max_date of the dataset
        enable_cold_start (bool): Enable cold start forecasting by training/predicting for zero history items.
        enable_multiple_backtests (bool): Whether to enable multiple backtesting or not.
        num_backtesting_windows (int): Total backtesting windows to use for the training.
        backtesting_window_step_size (int): Use this step size to shift backtesting windows for model training.
        full_data_retraining (bool): Train models separately with all the data.
        additional_forecast_keys: List[str]: List of categoricals in timeseries that can act as multi-identifier.
        experimentation_mode (ExperimentationMode): Selecting Thorough Experimentation will take longer to train.
        type_of_split (ForecastingDataSplitType): Type of data splitting into train/test.
        test_by_item (bool): Partition train/test data by item rather than time if true.
        test_start (str): Limit training data to dates before the given test start.
        test_split (int): Percent of dataset to use for test data. We support using a range between 5% to 20% of your dataset to use as test data.
        loss_function (ForecastingLossFunction): Loss function for training neural network.
        underprediction_weight (float): Weight for underpredictions
        disable_networks_without_analytic_quantiles (bool): Disable neural networks, which quantile functions do not have analytic expressions (e.g, mixture models)
        initial_learning_rate (float): Initial learning rate.
        l2_regularization_factor (float): L2 regularization factor.
        dropout_rate (int): Dropout percentage rate.
        recurrent_layers (int): Number of recurrent layers to stack in network.
        recurrent_units (int): Number of units in each recurrent layer.
        convolutional_layers (int): Number of convolutional layers to stack on top of recurrent layers in network.
        convolution_filters (int): Number of filters in each convolution.
        local_scaling_mode (ForecastingLocalScaling): Options to make NN inputs stationary in high dynamic range datasets.
        zero_predictor (bool): Include subnetwork to classify points where target equals zero.
        skip_missing (bool): Make the RNN ignore missing entries rather instead of processing them.
        batch_size (ForecastingBatchSize): Batch size.
        batch_renormalization (bool): Enable batch renormalization between layers.
        history_length (int): While training, how much history to consider.
        prediction_step_size (int): Number of future periods to include in objective for each training sample.
        training_point_overlap (float): Amount of overlap to allow between training samples.
        max_scale_context (int): Maximum context to use for local scaling.
        quantiles_extension_method (ForecastingQuanitlesExtensionMethod): Quantile extension method
        number_of_samples (int): Number of samples for ancestral simulation
        symmetrize_quantiles (bool): Force symmetric quantiles (like in Gaussian distribution)
        use_log_transforms (bool): Apply logarithmic transformations to input data.
        smooth_history (float): Smooth (low pass filter) the timeseries.
        local_scale_target (bool): Using per training/prediction window target scaling.
        use_clipping (bool): Apply clipping to input data to stabilize the training.
        timeseries_weight_column (str): If set, we use the values in this column from timeseries data to assign time dependent item weights during training and evaluation.
        item_attributes_weight_column (str): If set, we use the values in this column from item attributes data to assign weights to items during training and evaluation.
        use_timeseries_weights_in_objective (bool): If True, we include weights from column set as "TIMESERIES WEIGHT COLUMN" in objective functions.
        use_item_weights_in_objective (bool): If True, we include weights from column set as "ITEM ATTRIBUTES WEIGHT COLUMN" in objective functions.
        skip_timeseries_weight_scaling (bool): If True, we will avoid normalizing the weights.
        timeseries_loss_weight_column (str): Use value in this column to weight the loss while training.
        use_item_id (bool): Include a feature to indicate the item being forecast.
        use_all_item_totals (bool): Include as input total target across items.
        handle_zeros_as_missing_values (bool): If True, handle zero values in demand as missing data.
        datetime_holiday_calendars (List[HolidayCalendars]): Holiday calendars to augment training with.
        fill_missing_values (List[List[dict]]): Strategy for filling in missing values.
        enable_clustering (bool): Enable clustering in forecasting.
        data_split_feature_group_table_name (str): Specify the table name of the feature group to export training data with the fold column.
        custom_loss_functions (List[str]): Registered custom losses available for selection.
        custom_metrics (List[str]): Registered custom metrics available for selection.
        return_fractional_forecasts: Use this to return fractional forecast values while prediction
        allow_training_with_small_history: Allows training with fewer than 100 rows in the dataset
    """
    prediction_length: int = dataclasses.field(default=None)
    objective: enums.ForecastingObjective = dataclasses.field(default=None)
    sort_objective: enums.ForecastingObjective = dataclasses.field(default=None)
    forecast_frequency: enums.ForecastingFrequency = dataclasses.field(default=None)
    probability_quantiles: List[float] = dataclasses.field(default=None, metadata={'aichat': 'If None, defaults to [0.1, 0.5, 0.9]. If specified, then that list of quantiles will be used. You usually want to include the defaults.'})
    force_prediction_length: bool = dataclasses.field(default=None)
    filter_items: bool = dataclasses.field(default=None)
    enable_feature_selection: bool = dataclasses.field(default=None)
    enable_padding: bool = dataclasses.field(default=None)
    enable_cold_start: bool = dataclasses.field(default=None)
    enable_multiple_backtests: bool = dataclasses.field(default=None)
    num_backtesting_windows: int = dataclasses.field(default=None)
    backtesting_window_step_size: int = dataclasses.field(default=None)
    full_data_retraining: bool = dataclasses.field(default=None)
    additional_forecast_keys: List[str] = dataclasses.field(default=None)
    experimentation_mode: enums.ExperimentationMode = dataclasses.field(default=None)
    # Data split params
    type_of_split: enums.ForecastingDataSplitType = dataclasses.field(default=None)
    test_by_item: bool = dataclasses.field(default=None)
    test_start: str = dataclasses.field(default=None)
    test_split: int = dataclasses.field(default=None)
    # Neural network
    loss_function: enums.ForecastingLossFunction = dataclasses.field(default=None)
    underprediction_weight: float = dataclasses.field(default=None)
    disable_networks_without_analytic_quantiles: bool = dataclasses.field(default=None)
    initial_learning_rate: float = dataclasses.field(default=None)
    l2_regularization_factor: float = dataclasses.field(default=None)
    dropout_rate: int = dataclasses.field(default=None)
    recurrent_layers: int = dataclasses.field(default=None)
    recurrent_units: int = dataclasses.field(default=None)
    convolutional_layers: int = dataclasses.field(default=None)
    convolution_filters: int = dataclasses.field(default=None)
    local_scaling_mode: enums.ForecastingLocalScaling = dataclasses.field(default=None)
    zero_predictor: bool = dataclasses.field(default=None)
    skip_missing: bool = dataclasses.field(default=None)
    batch_size: enums.BatchSize = dataclasses.field(default=None)
    batch_renormalization: bool = dataclasses.field(default=None)
    # Timeseries
    history_length: int = dataclasses.field(default=None)
    prediction_step_size: int = dataclasses.field(default=None)
    training_point_overlap: float = dataclasses.field(default=None)
    max_scale_context: int = dataclasses.field(default=None)
    quantiles_extension_method: enums.ForecastingQuanitlesExtensionMethod = dataclasses.field(default=None)
    number_of_samples: int = dataclasses.field(default=None)
    symmetrize_quantiles: bool = dataclasses.field(default=None)
    use_log_transforms: bool = dataclasses.field(default=None)
    smooth_history: float = dataclasses.field(default=None)
    local_scale_target: bool = dataclasses.field(default=None)
    use_clipping: bool = dataclasses.field(default=None)
    # Item weights
    timeseries_weight_column: str = dataclasses.field(default=None)
    item_attributes_weight_column: str = dataclasses.field(default=None)
    use_timeseries_weights_in_objective: bool = dataclasses.field(default=None)
    use_item_weights_in_objective: bool = dataclasses.field(default=None)
    skip_timeseries_weight_scaling: bool = dataclasses.field(default=None)
    timeseries_loss_weight_column: str = dataclasses.field(default=None)
    # Data Augmentation
    use_item_id: bool = dataclasses.field(default=None)
    use_all_item_totals: bool = dataclasses.field(default=None)
    handle_zeros_as_missing_values: bool = dataclasses.field(default=None)
    datetime_holiday_calendars: List[enums.HolidayCalendars] = dataclasses.field(default=None)
    fill_missing_values: List[List[dict]] = dataclasses.field(default=None)
    enable_clustering: bool = dataclasses.field(default=None)
    # Others
    data_split_feature_group_table_name: str = dataclasses.field(default=None)
    custom_loss_functions: List[str] = dataclasses.field(default=None)
    custom_metrics: List[str] = dataclasses.field(default=None)
    return_fractional_forecasts: bool = dataclasses.field(default=None)
    allow_training_with_small_history: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.FORECASTING


@dataclasses.dataclass
class NamedEntityExtractionTrainingConfig(TrainingConfig):
    """
    Training config for the NAMED_ENTITY_EXTRACTION problem type

    Args:
        llm_for_ner (NERForLLM) : LLM to use for NER from among available LLM
        test_split (int): Percent of dataset to use for test data. We support using a range between 5 ( i.e. 5% ) to 20 ( i.e. 20% ) of your dataset.
        test_row_indicator (str): Column indicating which rows to use for training (TRAIN) and testing (TEST).
        active_labels_column (str): Entities that have been marked in a particular text
        document_format (NLPDocumentFormat): Format of the input documents.
        minimum_bounding_box_overlap_ratio (float): Tokens are considered to belong to annotation if the user bounding box is provided and ratio of (token_bounding_box ∩ annotation_bounding_box) / token_bounding_area is greater than the provided value.
        save_predicted_pdf (bool): Whether to save predicted PDF documents
        enhanced_ocr (bool): Enhanced text extraction from predicted digital documents
        additional_extraction_instructions (str): Additional instructions to guide the LLM in extracting the entities. Only used with LLM algorithms.
    """
    llm_for_ner: enums.LLMName = None
    # Data Split Params
    test_split: int = None
    test_row_indicator: str = None
    # Named Entity Recognition
    active_labels_column: str = None
    document_format: enums.NLPDocumentFormat = None
    minimum_bounding_box_overlap_ratio: float = 0.0
    # OCR
    save_predicted_pdf: bool = True
    enhanced_ocr: bool = False
    additional_extraction_instructions: str = None

    def __post_init__(self):
        self.problem_type = enums.ProblemType.NAMED_ENTITY_EXTRACTION


@dataclasses.dataclass
class NaturalLanguageSearchTrainingConfig(TrainingConfig):
    """
    Training config for the NATURAL_LANGUAGE_SEARCH problem type

    Args:
        abacus_internal_model (bool): Use a Abacus.AI LLM to answer questions about your data without using any external APIs
        num_completion_tokens (int): Default for maximum number of tokens for chat answers. Reducing this will get faster responses which are more succinct
        larger_embeddings (bool): Use a higher dimension embedding model.
        search_chunk_size (int): Chunk size for indexing the documents.
        chunk_overlap_fraction (float): Overlap in chunks while indexing the documents.
        index_fraction (float): Fraction of the chunk to use for indexing.
    """
    abacus_internal_model: bool = dataclasses.field(default=None)
    num_completion_tokens: int = dataclasses.field(default=None)
    larger_embeddings: bool = dataclasses.field(default=None)
    search_chunk_size: int = dataclasses.field(default=None)
    index_fraction: float = dataclasses.field(default=None)
    chunk_overlap_fraction: float = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.NATURAL_LANGUAGE_SEARCH


@dataclasses.dataclass
class ChatLLMTrainingConfig(TrainingConfig):
    """
    Training config for the CHAT_LLM problem type

    Args:
        document_retrievers (List[str]): List of names or IDs of document retrievers to use as vector stores of information for RAG responses.
        num_completion_tokens (int): Default for maximum number of tokens for chat answers. Reducing this will get faster responses which are more succinct.
        temperature (float): The generative LLM temperature.
        retrieval_columns (list): Include the metadata column values in the retrieved search results.
        filter_columns (list): Allow users to filter the document retrievers on these metadata columns.
        include_general_knowledge (bool): Allow the LLM to rely not just on RAG search results, but to fall back on general knowledge. Disabled by default.
        enable_web_search (bool) : Allow the LLM to use Web Search Engines to retrieve information for better results.
        behavior_instructions (str): Customize the overall behaviour of the model. This controls things like - when to execute code (if enabled), write sql query, search web (if enabled), etc.
        response_instructions (str): Customized instructions for how the model should respond inlcuding the format, persona and tone of the answers.
        enable_llm_rewrite (bool): If enabled, an LLM will rewrite the RAG queries sent to document retriever. Disabled by default.
        column_filtering_instructions (str): Instructions for a LLM call to automatically generate filter expressions on document metadata to retrieve relevant documents for the conversation.
        keyword_requirement_instructions (str): Instructions for a LLM call to automatically generate keyword requirements to retrieve relevant documents for the conversation.
        query_rewrite_instructions (str): Special instructions for the LLM which rewrites the RAG query.
        max_search_results (int): Maximum number of search results in the retrieval augmentation step. If we know that the questions are likely to have snippets which are easily matched in the documents, then a lower number will help with accuracy.
        data_feature_group_ids: (List[str]): List of feature group IDs to use to possibly query for the ChatLLM. The created ChatLLM is commonly referred to as DataLLM.
        data_prompt_context (str): Prompt context for the data feature group IDs.
        data_prompt_table_context (Dict[str, str]): Dict of table name and table context pairs to provide table wise context for each structured data table.
        data_prompt_column_context (Dict[str, str]): Dict of 'table_name.column_name' and 'column_context' pairs to provide column context for some selected columns in the selected structured data table. This replaces the default auto-generated information about the column data.
        hide_sql_and_code (bool): When running data queries, this will hide the generated SQL and Code in the response.
        disable_data_summarization (bool): After executing a query summarize the reponse and reply back with only the table and query run.
        data_columns_to_ignore (List[str]): Columns to ignore while encoding information about structured data tables in context for the LLM. A list of strings of format "<table_name>.<column_name>"
        search_score_cutoff (float): Minimum search score to consider a document as a valid search result.
        include_bm25_retrieval (bool): Combine BM25 search score with vector search using reciprocal rank fusion.
        database_connector_id (str): Database connector ID to use for connecting external database that gives access to structured data to the LLM.
        database_connector_tables (List[str]): List of tables to use from the database connector for the ChatLLM.
        enable_code_execution (bool): Enable python code execution in the ChatLLM. This equips the LLM with a python kernel in which all its code is executed.
        enable_response_caching (bool): Enable caching of LLM responses to speed up response times and improve reproducibility.
        unknown_answer_phrase (str): Fallback response when the LLM can't find an answer.
        enable_tool_bar (bool): Enable the tool bar in Enterprise ChatLLM to provide additional functionalities like tool_use, web_search, image_gen, etc.
        enable_inline_source_citations (bool): Enable inline citations of the sources in the response.
        response_format: (str): When set to 'JSON', the LLM will generate a JSON formatted string.
        json_response_instructions (str): Instructions to be followed while generating the json_response if `response_format` is set to "JSON". This can include the schema information if the schema is dynamic and its keys cannot be pre-determined.
        json_response_schema (str): Specifies the JSON schema that the model should adhere to if `response_format` is set to "JSON". This should be a json-formatted string where each field of the expected schema is mapped to a dictionary containing the fields 'type', 'required' and 'description'. For example - '{"sample_field": {"type": "integer", "required": true, "description": "Sample Field"}}'
        mask_pii (bool): Mask PII in the prompts and uploaded documents before sending it to the LLM.
    """
    document_retrievers: List[str] = dataclasses.field(default=None)
    num_completion_tokens: int = dataclasses.field(default=None)
    temperature: float = dataclasses.field(default=None)
    retrieval_columns: list = dataclasses.field(default=None)
    filter_columns: list = dataclasses.field(default=None)
    include_general_knowledge: bool = dataclasses.field(default=None)
    enable_web_search: bool = dataclasses.field(default=None)
    behavior_instructions: str = dataclasses.field(default=None)
    response_instructions: str = dataclasses.field(default=None)
    enable_llm_rewrite: bool = dataclasses.field(default=None)
    column_filtering_instructions: str = dataclasses.field(default=None)
    keyword_requirement_instructions: str = dataclasses.field(default=None)
    query_rewrite_instructions: str = dataclasses.field(default=None)
    max_search_results: int = dataclasses.field(default=None)
    data_feature_group_ids: List[str] = dataclasses.field(default=None)
    data_prompt_context: str = dataclasses.field(default=None)
    data_prompt_table_context: Dict[str, str] = dataclasses.field(default=None)
    data_prompt_column_context: Dict[str, str] = dataclasses.field(default=None)
    hide_sql_and_code: bool = dataclasses.field(default=None)
    disable_data_summarization: bool = dataclasses.field(default=None)
    data_columns_to_ignore: List[str] = dataclasses.field(default=None)
    search_score_cutoff: float = dataclasses.field(default=None)
    include_bm25_retrieval: bool = dataclasses.field(default=None)
    database_connector_id: str = dataclasses.field(default=None)
    database_connector_tables: List[str] = dataclasses.field(default=None)
    enable_code_execution: bool = dataclasses.field(default=None)
    metadata_columns: list = dataclasses.field(default=None, metadata={'deprecated': True})
    lookup_rewrite_instructions: str = dataclasses.field(default=None, metadata={'deprecated': True})
    enable_response_caching: bool = dataclasses.field(default=None)
    unknown_answer_phrase: str = dataclasses.field(default=None)
    enable_tool_bar: bool = dataclasses.field(default=None)
    enable_inline_source_citations: bool = dataclasses.field(default=None)
    response_format: str = dataclasses.field(default=None)
    json_response_instructions: str = dataclasses.field(default=None)
    json_response_schema: str = dataclasses.field(default=None)
    mask_pii: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.CHAT_LLM


@dataclasses.dataclass
class SentenceBoundaryDetectionTrainingConfig(TrainingConfig):
    """
    Training config for the SENTENCE_BOUNDARY_DETECTION problem type

    Args:
        test_split (int): Percent of dataset to use for test data. We support using a range between 5 ( i.e. 5% ) to 20 ( i.e. 20% ) of your dataset.
        dropout_rate (float): Dropout rate for neural network.
        batch_size (BatchSize): Batch size for neural network.
    """
    # Data Split Params
    test_split: int = dataclasses.field(default=None)
    # Neural Network
    dropout_rate: float = dataclasses.field(default=None)
    batch_size: enums.BatchSize = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.SENTENCE_BOUNDARY_DETECTION


@dataclasses.dataclass
class SentimentDetectionTrainingConfig(TrainingConfig):
    """
    Training config for the SENTIMENT_DETECTION problem type

    Args:
        sentiment_type (SentimentType): Type of sentiment to detect.
        test_split (int): Percent of dataset to use for test data. We support using a range between 5 ( i.e. 5% ) to 20 ( i.e. 20% ) of your dataset.
    """
    sentiment_type: enums.SentimentType = dataclasses.field(default=None)
    # Data Split Params
    test_split: int = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.SENTIMENT_DETECTION


@dataclasses.dataclass
class DocumentClassificationTrainingConfig(TrainingConfig):
    """
    Training config for the DOCUMENT_CLASSIFICATION problem type

    Args:
        zero_shot_hypotheses (List[str]): Zero shot hypotheses. Example text: 'This text is about pricing'.
        test_split (int): Percent of dataset to use for test data. We support using a range between 5 ( i.e. 5% ) to 20 ( i.e. 20% ) of your dataset.
    """
    zero_shot_hypotheses: List[str] = dataclasses.field(default=None)
    # Data Split Params
    test_split: int = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.DOCUMENT_CLASSIFICATION


@dataclasses.dataclass
class DocumentSummarizationTrainingConfig(TrainingConfig):
    """
    Training config for the DOCUMENT_SUMMARIZATION problem type

    Args:
        test_split (int): Percent of dataset to use for test data. We support using a range between 5 ( i.e. 5% ) to 20 ( i.e. 20% ) of your dataset.
        dropout_rate (float): Dropout rate for neural network.
        batch_size (BatchSize): Batch size for neural network.
    """
    # Data Split Params
    test_split: int = dataclasses.field(default=None)
    # Neural Network
    dropout_rate: float = dataclasses.field(default=None)
    batch_size: enums.BatchSize = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.DOCUMENT_SUMMARIZATION


@dataclasses.dataclass
class DocumentVisualizationTrainingConfig(TrainingConfig):
    """
    Training config for the DOCUMENT_VISUALIZATION problem type

    Args:
        test_split (int): Percent of dataset to use for test data. We support using a range between 5 ( i.e. 5% ) to 20 ( i.e. 20% ) of your dataset.
        dropout_rate (float): Dropout rate for neural network.
        batch_size (BatchSize): Batch size for neural network.
    """
    # Data Split Params
    test_split: int = dataclasses.field(default=None)
    # Neural Network
    dropout_rate: float = dataclasses.field(default=None)
    batch_size: enums.BatchSize = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.DOCUMENT_VISUALIZATION


@dataclasses.dataclass
class ClusteringTrainingConfig(TrainingConfig):
    """
    Training config for the CLUSTERING problem type

    Args:
        num_clusters_selection (int): Number of clusters. If None, will be selected automatically.
    """
    num_clusters_selection: int = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.CLUSTERING


@dataclasses.dataclass
class ClusteringTimeseriesTrainingConfig(TrainingConfig):
    """
    Training config for the CLUSTERING_TIMESERIES problem type

    Args:
        num_clusters_selection (int): Number of clusters. If None, will be selected automatically.
        imputation (ClusteringImputationMethod): Imputation method for missing values.
    """
    num_clusters_selection: int = dataclasses.field(default=None)
    imputation: enums.ClusteringImputationMethod = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.CLUSTERING_TIMESERIES


@dataclasses.dataclass
class EventAnomalyTrainingConfig(TrainingConfig):
    """
    Training config for the EVENT_ANOMALY problem type

    Args:
        anomaly_fraction (float): The fraction of the dataset to classify as anomalous, between 0 and 0.5
    """
    anomaly_fraction: float = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.EVENT_ANOMALY


@dataclasses.dataclass
class TimeseriesAnomalyTrainingConfig(TrainingConfig):
    """
    Training config for the TS_ANOMALY problem type

    Args:
        type_of_split (TimeseriesAnomalyDataSplitType): Type of data splitting into train/test.
        test_start (str): Limit training data to dates before the given test start.
        test_split (int): Percent of dataset to use for test data. We support using a range between 5 ( i.e. 5% ) to 20 ( i.e. 20% ) of your dataset.
        fill_missing_values (List[List[dict]]): strategies to fill missing values and missing timestamps
        handle_zeros_as_missing_values (bool): If True, handle zero values in numeric columns as missing data
        timeseries_frequency (str): set this to control frequency of filling missing values
        min_samples_in_normal_region (int): Adjust this to fine-tune the number of anomalies to be identified.
        anomaly_type (TimeseriesAnomalyTypeOfAnomaly): select what kind of peaks to detect as anomalies
        hyperparameter_calculation_with_heuristics (TimeseriesAnomalyUseHeuristic): Enable heuristic calculation to get hyperparameters for the model
        threshold_score (float): Threshold score for anomaly detection
        additional_anomaly_ids (List[str]): List of categorical columns that can act as multi-identifier
    """
    type_of_split: enums.TimeseriesAnomalyDataSplitType = dataclasses.field(default=None)
    test_start: str = dataclasses.field(default=None)
    test_split: int = dataclasses.field(default=None)
    fill_missing_values: List[List[dict]] = dataclasses.field(default=None)
    handle_zeros_as_missing_values: bool = dataclasses.field(default=None)
    timeseries_frequency: str = dataclasses.field(default=None)
    min_samples_in_normal_region: int = dataclasses.field(default=None)
    anomaly_type: enums.TimeseriesAnomalyTypeOfAnomaly = dataclasses.field(default=None)
    hyperparameter_calculation_with_heuristics: enums.TimeseriesAnomalyUseHeuristic = dataclasses.field(default=None)
    threshold_score: float = dataclasses.field(default=None)
    additional_anomaly_ids: List[str] = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.TS_ANOMALY


@dataclasses.dataclass
class CumulativeForecastingTrainingConfig(TrainingConfig):
    """
    Training config for the CUMULATIVE_FORECASTING problem type

    Args:
        test_split (int): Percent of dataset to use for test data. We support using a range between 5 ( i.e. 5% ) to 20 ( i.e. 20% ) of your dataset.
        historical_frequency (str): Forecast frequency
        cumulative_prediction_lengths (List[int]): List of Cumulative Prediction Frequencies. Each prediction length must be between 1 and 365.
        skip_input_transform (bool): Avoid doing numeric scaling transformations on the input.
        skip_target_transform (bool): Avoid doing numeric scaling transformations on the target.
        predict_residuals (bool): Predict residuals instead of totals at each prediction step.
    """
    test_split: int = dataclasses.field(default=None)
    historical_frequency: str = dataclasses.field(default=None)
    cumulative_prediction_lengths: List[int] = dataclasses.field(default=None)
    skip_input_transform: bool = dataclasses.field(default=None)
    skip_target_transform: bool = dataclasses.field(default=None)
    predict_residuals: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.CUMULATIVE_FORECASTING


@dataclasses.dataclass
class ThemeAnalysisTrainingConfig(TrainingConfig):
    """
    Training config for the THEME ANALYSIS problem type
    """

    def __post_init__(self):
        self.problem_type = enums.ProblemType.THEME_ANALYSIS


@dataclasses.dataclass
class AIAgentTrainingConfig(TrainingConfig):
    """
    Training config for the AI_AGENT problem type

    Args:
        description (str): Description of the agent function.
        agent_interface (AgentInterface): The interface that the agent will be deployed with.
        agent_connectors: (List[enums.ApplicationConnectorType]): The connectors needed for the agent to function.
    """
    description: str = dataclasses.field(default=None)
    agent_interface: enums.AgentInterface = dataclasses.field(default=None)
    agent_connectors: List[enums.ApplicationConnectorType] = dataclasses.field(default=None)
    enable_binary_input: bool = dataclasses.field(default=None, metadata={'deprecated': True})
    agent_input_schema: dict = dataclasses.field(default=None, metadata={'deprecated': True})
    agent_output_schema: dict = dataclasses.field(default=None, metadata={'deprecated': True})

    def __post_init__(self):
        self.problem_type = enums.ProblemType.AI_AGENT


@dataclasses.dataclass
class CustomTrainedModelTrainingConfig(TrainingConfig):
    """
    Training config for the CUSTOM_TRAINED_MODEL problem type

    Args:
        max_catalog_size (int): Maximum expected catalog size.
        max_dimension (int): Maximum expected dimension of the catalog.
        index_output_path (str): Fully qualified cloud location (GCS, S3, etc) to export snapshots of the embedding to.
        docker_image_uri (str): Docker image URI.
        service_port (int): Service port.
        streaming_embeddings (bool): Flag to enable streaming embeddings.
    """
    max_catalog_size: int = dataclasses.field(default=None)
    max_dimension: int = dataclasses.field(default=None)
    index_output_path: str = dataclasses.field(default=None)
    docker_image_uri: str = dataclasses.field(default=None)
    service_port: int = dataclasses.field(default=None)
    streaming_embeddings: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.CUSTOM_TRAINED_MODEL


@dataclasses.dataclass
class CustomAlgorithmTrainingConfig(TrainingConfig):
    """
    Training config for the CUSTOM_ALGORITHM problem type

    Args:
        timeout_minutes (int): Timeout for the model training in minutes.
    """
    timeout_minutes: int = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.CUSTOM_ALGORITHM


@dataclasses.dataclass
class OptimizationTrainingConfig(TrainingConfig):
    """
    Training config for the OPTIMIZATION problem type

    Args:
        solve_time_limit (float): The maximum time in seconds to spend solving the problem. Accepts values between 0 and 86400.
        optimality_gap_limit (float): The stopping optimality gap limit. Optimality gap is fractional difference between the best known solution and the best possible solution. Accepts values between 0 and 1.
        include_all_partitions (bool): Include all partitions in the model training. Default is False.
        include_specific_partitions (List[str]): Include specific partitions in partitioned model training. Default is empty list.
    """
    solve_time_limit: float = dataclasses.field(default=None)
    optimality_gap_limit: float = dataclasses.field(default=None)
    include_all_partitions: bool = dataclasses.field(default=None)
    include_specific_partitions: List[str] = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.OPTIMIZATION


@dataclasses.dataclass
class _TrainingConfigFactory(_ApiClassFactory):
    config_abstract_class = TrainingConfig
    config_class_key = 'problem_type'
    config_class_map = {
        enums.ProblemType.AI_AGENT: AIAgentTrainingConfig,
        enums.ProblemType.CLUSTERING: ClusteringTrainingConfig,
        enums.ProblemType.CLUSTERING_TIMESERIES: ClusteringTimeseriesTrainingConfig,
        enums.ProblemType.CUMULATIVE_FORECASTING: CumulativeForecastingTrainingConfig,
        enums.ProblemType.CUSTOM_TRAINED_MODEL: CustomTrainedModelTrainingConfig,
        enums.ProblemType.DOCUMENT_CLASSIFICATION: DocumentClassificationTrainingConfig,
        enums.ProblemType.DOCUMENT_SUMMARIZATION: DocumentSummarizationTrainingConfig,
        enums.ProblemType.DOCUMENT_VISUALIZATION: DocumentVisualizationTrainingConfig,
        enums.ProblemType.EVENT_ANOMALY: EventAnomalyTrainingConfig,
        enums.ProblemType.FORECASTING: ForecastingTrainingConfig,
        enums.ProblemType.NAMED_ENTITY_EXTRACTION: NamedEntityExtractionTrainingConfig,
        enums.ProblemType.NATURAL_LANGUAGE_SEARCH: NaturalLanguageSearchTrainingConfig,
        enums.ProblemType.CHAT_LLM: ChatLLMTrainingConfig,
        enums.ProblemType.PERSONALIZATION: PersonalizationTrainingConfig,
        enums.ProblemType.PREDICTIVE_MODELING: RegressionTrainingConfig,
        enums.ProblemType.SENTENCE_BOUNDARY_DETECTION: SentenceBoundaryDetectionTrainingConfig,
        enums.ProblemType.SENTIMENT_DETECTION: SentimentDetectionTrainingConfig,
        enums.ProblemType.THEME_ANALYSIS: ThemeAnalysisTrainingConfig,
        enums.ProblemType.CUSTOM_ALGORITHM: CustomAlgorithmTrainingConfig,
        enums.ProblemType.OPTIMIZATION: OptimizationTrainingConfig,
        enums.ProblemType.TS_ANOMALY: TimeseriesAnomalyTrainingConfig,
    }


@dataclasses.dataclass
class DeployableAlgorithm(ApiClass):
    """
    Algorithm that can be deployed to a model.

    Args:
        algorithm (str): ID of the algorithm.
        name (str): Name of the algorithm.
        only_offline_deployable (bool): Whether the algorithm can only be deployed offline.
        trained_model_types (List[dict]): List of trained model types.
    """
    algorithm: str = dataclasses.field(default=None)
    name: str = dataclasses.field(default=None)
    only_offline_deployable: bool = dataclasses.field(default=None)
    trained_model_types: List[dict] = dataclasses.field(default=None)
