import dataclasses
from typing import Dict, List

from . import enums
from .abstract import ApiClass, _ApiClassFactory


@dataclasses.dataclass
class TrainingConfig(ApiClass):
    _upper_snake_case_keys: bool = dataclasses.field(default=True, repr=False, init=False)
    _support_kwargs: bool = dataclasses.field(default=True, repr=False, init=False)

    kwargs: dict = dataclasses.field(default_factory=dict)
    problem_type: enums.ProblemType = dataclasses.field(default=None, repr=False, init=False)
    algorithm: str = dataclasses.field(default=None)


@dataclasses.dataclass
class PersonalizationTrainingConfig(TrainingConfig):
    """
    Training config for the PERSONALIZATION problem type
    Args:
        objective (PersonalizationObjective): Ranking scheme used to select final best model.
        sort_objective (PersonalizationObjective): Ranking scheme used to sort models on the metrics page.
        training_mode (PersonalizationTrainingMode): whether to train in production or experimental mode.
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
        dropout_rate (int): Dropout rate for neural network.
        batch_size (BatchSize): Batch size for neural network.
        disable_transformer (bool): Disable training the transformer algorithm.
        disable_gpu (boo): Disable training on GPU.
        filter_history (bool): Do not recommend items the user has already interacted with.
        explore_lookback_hours (int): Number of hours since creation time that an item is eligible for explore fraction.
        max_history_length (int): Maximum length of user-item history to include user in training examples.
        compute_rerank_metrics (bool): Compute metrics based on rerank results.
        item_id_dropout (float): Fraction of item_id values to randomly dropout during training.
        add_time_features (bool): Include interaction time as a feature.
        disable_timestamp_scalar_features (bool): Exclude timestamp scalar features.
        compute_session_metrics (bool): Evaluate models based on how well they are able to predict the next session of interactions.
        max_user_history_len_percentile (int): Filter out users with history length above this percentile.
        downsample_item_popularity_percentile (float): Downsample items more popular than this percentile.

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

    # neural network
    dropout_rate: int = dataclasses.field(default=None)
    batch_size: enums.BatchSize = dataclasses.field(default=None)
    disable_transformer: bool = dataclasses.field(default=None)
    disable_gpu: bool = dataclasses.field(default=None)

    # prediction
    filter_history: bool = dataclasses.field(default=None)
    explore_lookback_hours: int = dataclasses.field(default=None)

    # data distribution
    max_history_length: int = dataclasses.field(default=None)
    compute_rerank_metrics: bool = dataclasses.field(default=None)
    item_id_dropout: float = dataclasses.field(default=None)
    add_time_features: bool = dataclasses.field(default=None)
    disable_timestamp_scalar_features: bool = dataclasses.field(default=None)
    compute_session_metrics: bool = dataclasses.field(default=None)

    # outliers
    max_user_history_len_percentile: int = dataclasses.field(default=None)
    downsample_item_popularity_percentile: float = dataclasses.field(default=None)

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
        rebalance_classes (bool): Class weights are computed as the inverse of the class frequency from the training dataset when this option is selected as "Yes". It is useful when the classes in the dataset are unbalanced.
                                  Re-balancing classes generally boosts recall at the cost of precision on rare classes.
        rare_class_augmentation_threshold (float): Augments any rare class whose relative frequency with respect to the most frequent class is less than this threshold. Default = 0.1
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
        is_multilingual (bool): Enable algorithms which process text using pretrained multilingual NLP models.
        loss_function (RegressionLossFunction): Loss function to be used as objective for model training.
        loss_parameters (str): Loss function params in format <key>=<value>;<key>=<value>;.....
        target_encode_categoricals (bool): Use this to turn target encoding on categorical features on or off.
        drop_original_categoricals (bool): This option helps us choose whether to also feed the original label encoded categorical columns to the mdoels along with their target encoded versions.
        data_split_feature_group_table_name (str): Specify the table name of the feature group to export training data with the fold column.
        custom_loss_functions (List[str]): Registered custom losses available for selection.
        custom_metrics (List[str]): Registered custom metrics available for selection.

    """
    objective: enums.RegressionObjective = dataclasses.field(default=None)
    sort_objective: enums.RegressionObjective = dataclasses.field(default=None)
    tree_hpo_mode: enums.RegressionTreeHPOMode = dataclasses.field(default=enums.RegressionTreeHPOMode.THOROUGH)

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

    # data augmentation
    rebalance_classes: bool = dataclasses.field(default=None)
    rare_class_augmentation_threshold: float = dataclasses.field(default=0.1)
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
    is_multilingual: bool = dataclasses.field(default=None)

    # loss function
    loss_function: enums.RegressionLossFunction = dataclasses.field(default=None)
    loss_parameters: str = dataclasses.field(default=None)

    # target encoding
    target_encode_categoricals: bool = dataclasses.field(default=None)
    drop_original_categoricals: bool = dataclasses.field(default=None)

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
        skip_local_scale_target (bool): Skip using per training/prediction window target scaling.
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
        fill_missing_values (List[dict]): Strategy for filling in missing values.
        enable_clustering (bool): Enable clustering in forecasting.
        data_split_feature_group_table_name (str): Specify the table name of the feature group to export training data with the fold column.
        custom_loss_functions (List[str]): Registered custom losses available for selection.
        custom_metrics (List[str]): Registered custom metrics available for selection.
    """
    prediction_length: int = dataclasses.field(default=None)
    objective: enums.ForecastingObjective = dataclasses.field(default=None)
    sort_objective: enums.ForecastingObjective = dataclasses.field(default=None)
    forecast_frequency: enums.ForecastingFrequency = dataclasses.field(default=None)
    probability_quantiles: List[float] = dataclasses.field(default=None, metadata={'aichat': 'If None, defaults to [0.1, 0.5, 0.9]. If specified, then that list of quantiles will be used. You usually want to include the defaults.'})
    force_prediction_length: bool = dataclasses.field(default=None)
    filter_items: bool = dataclasses.field(default=None)
    enable_feature_selection: bool = dataclasses.field(default=None)
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
    skip_local_scale_target: bool = dataclasses.field(default=None)
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
    fill_missing_values: List[dict] = dataclasses.field(default=None)
    enable_clustering: bool = dataclasses.field(default=None)
    # Others
    data_split_feature_group_table_name: str = dataclasses.field(default=None)
    custom_loss_functions: List[str] = dataclasses.field(default=None)
    custom_metrics: List[str] = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.FORECASTING


@dataclasses.dataclass
class NamedEntityExtractionTrainingConfig(TrainingConfig):
    """
    Training config for the NAMED_ENTITY_EXTRACTION problem type
    Args:
        objective (NERObjective): Ranking scheme used to select final best model.
        sort_objective (NERObjective): Ranking scheme used to sort models on the metrics page.
        ner_model_type (NERModelType): Type of NER model to use.
        test_split (int): Percent of dataset to use for test data. We support using a range between 5 ( i.e. 5% ) to 20 ( i.e. 20% ) of your dataset.
        test_row_indicator (str): Column indicating which rows to use for training (TRAIN) and testing (TEST).
        dropout_rate (float): Dropout rate for neural network.
        batch_size (BatchSize): Batch size for neural network.
        active_labels_column (str): Entities that have been marked in a particular text
        document_format (NLPDocumentFormat): Format of the input documents.
        include_longformer (bool): Whether to include the longformer model.
    """
    objective: enums.NERObjective = dataclasses.field(default=None)
    sort_objective: enums.NERObjective = dataclasses.field(default=None)
    ner_model_type: enums.NERModelType = dataclasses.field(default=None)
    # Data Split Params
    test_split: int = dataclasses.field(default=None)
    test_row_indicator: str = dataclasses.field(default=None)
    # Neural Network
    dropout_rate: float = dataclasses.field(default=None)
    batch_size: enums.BatchSize = dataclasses.field(default=None)
    # Named Entity Recognition
    active_labels_column: str = dataclasses.field(default=None)
    document_format: enums.NLPDocumentFormat = dataclasses.field(default=None)
    include_longformer: bool = dataclasses.field(default=None)

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
        test_split (int): Percent of dataset to use for test data. We support using a range between 5 ( i.e. 5% ) to 20 ( i.e. 20% ) of your dataset.
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
        document_retrievers (List[str]): List of document retriever names to use for the feature stores this model was trained with.
        num_completion_tokens (int): Default for maximum number of tokens for chat answers. Reducing this will get faster responses which are more succinct
        system_message (str): The generative LLM system message
        temperature (float): The generative LLM temperature
        search_title_column (str): Include the title column values in the retrieved search results
    """
    document_retrievers: List[str] = None
    num_completion_tokens: int = None
    system_message: str = None
    temperature: float = None
    search_title_column: str = None

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
        dropout_rate (float): Dropout rate for neural network.
        batch_size (BatchSize): Batch size for neural network.
        compute_metrics (bool): Whether to compute metrics.
    """
    sentiment_type: enums.SentimentType = dataclasses.field(default=None)
    # Data Split Params
    test_split: int = dataclasses.field(default=None)
    # Neural Network
    dropout_rate: float = dataclasses.field(default=None)
    batch_size: enums.BatchSize = dataclasses.field(default=None)
    # Metrics
    compute_metrics: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.SENTIMENT_DETECTION


@dataclasses.dataclass
class DocumentClassificationTrainingConfig(TrainingConfig):
    """
    Training config for the DOCUMENT_CLASSIFICATION problem type
    Args:
        zero_shot_hypotheses (List[str]): Zero shot hypotheses. Example text: 'This text is about pricing'.
        test_split (int): Percent of dataset to use for test data. We support using a range between 5 ( i.e. 5% ) to 20 ( i.e. 20% ) of your dataset.
        dropout_rate (float): Dropout rate for neural network.
        batch_size (BatchSize): Batch size for neural network.
    """
    zero_shot_hypotheses: List[str] = dataclasses.field(default=None)
    # Data Split Params
    test_split: int = dataclasses.field(default=None)
    # Neural Network
    dropout_rate: float = dataclasses.field(default=None)
    batch_size: enums.BatchSize = dataclasses.field(default=None)

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
class AnomalyDetectionTrainingConfig(TrainingConfig):
    """
    Training config for the ANOMALY_DETECTION problem type
    Args:
        test_split (int): Percent of dataset to use for test data. We support using a range between 5 (i.e. 5%) to 20 (i.e. 20%) of your dataset to use as test data.
        value_high (bool): Detect unusually high values.
        mixture_of_gaussians (bool): Detect unusual combinations of values using mixture of Gaussians.
        variational_autoencoder (bool): Use variational autoencoder for anomaly detection.
        spike_up (bool): Detect outliers with a high value.
        spike_down (bool): Detect outliers with a low value.
        trend_change (bool): Detect changes to the trend.
    """
    test_split: int = dataclasses.field(default=None)
    value_high: bool = dataclasses.field(default=None)
    mixture_of_gaussians: bool = dataclasses.field(default=None)
    variational_autoencoder: bool = dataclasses.field(default=None)
    spike_up: bool = dataclasses.field(default=None)
    spike_down: bool = dataclasses.field(default=None)
    trend_change: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.ANOMALY_DETECTION


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
    """
    description: str = dataclasses.field(default=None)

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
    """
    max_catalog_size: int = dataclasses.field(default=None)
    max_dimension: int = dataclasses.field(default=None)
    index_output_path: str = dataclasses.field(default=None)
    docker_image_uri: str = dataclasses.field(default=None)
    service_port: int = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.CUSTOM_TRAINED_MODEL


@dataclasses.dataclass
class CustomAlgorithmTrainingConfig(TrainingConfig):
    """
    Training config for the CUSTOM_ALGORITHM problem type
    Args:
        train_function_name (str): The name of the train function.
        predict_many_function_name (str): The name of the predict many function.
        training_input_tables (List[str]): List of tables to use for training.
        predict_function_name (str): Optional name of the predict function if the predict many function is not given.
        train_module_name (str): The name of the train module - only relevant if model is being uploaded from a zip file or github repositoty.
        predict_module_name (str): The name of the predict module - only relevant if model is being uploaded from a zip file or github repositoty.
        test_split (int): Percent of dataset to use for test data. We support using a range between 6% to 20% of your dataset to use as test data.
    """
    train_function_name: str = dataclasses.field(default=None)
    predict_many_function_name: str = dataclasses.field(default=None)
    training_input_tables: List[str] = dataclasses.field(default=None)
    predict_function_name: str = dataclasses.field(default=None)
    train_module_name: str = dataclasses.field(default=None)
    predict_module_name: str = dataclasses.field(default=None)
    test_split: int = dataclasses.field(default=None)

    def __post_init__(self):
        self.problem_type = enums.ProblemType.CUSTOM_ALGORITHM


@dataclasses.dataclass
class OptimizationTrainingConfig(TrainingConfig):
    """
    Training config for the OPTIMIZATION problem type
    """

    def __post_init__(self):
        self.problem_type = enums.ProblemType.OPTIMIZATION


@dataclasses.dataclass
class _TrainingConfigFactory(_ApiClassFactory):
    config_abstract_class = TrainingConfig
    config_class_key = 'problem_type'
    config_class_map = {
        enums.ProblemType.AI_AGENT: AIAgentTrainingConfig,
        enums.ProblemType.ANOMALY_DETECTION: AnomalyDetectionTrainingConfig,
        enums.ProblemType.CLUSTERING: ClusteringTrainingConfig,
        enums.ProblemType.CLUSTERING_TIMESERIES: ClusteringTimeseriesTrainingConfig,
        enums.ProblemType.CUMULATIVE_FORECASTING: CumulativeForecastingTrainingConfig,
        enums.ProblemType.CUSTOM_TRAINED_MODEL: CustomTrainedModelTrainingConfig,
        enums.ProblemType.DOCUMENT_CLASSIFICATION: DocumentClassificationTrainingConfig,
        enums.ProblemType.DOCUMENT_SUMMARIZATION: DocumentSummarizationTrainingConfig,
        enums.ProblemType.DOCUMENT_VISUALIZATION: DocumentVisualizationTrainingConfig,
        enums.ProblemType.FORECASTING: ForecastingTrainingConfig,
        enums.ProblemType.NAMED_ENTITY_EXTRACTION: NamedEntityExtractionTrainingConfig,
        enums.ProblemType.NATURAL_LANGUAGE_SEARCH: NaturalLanguageSearchTrainingConfig,
        enums.ProblemType.CHAT_LLM: ChatLLMTrainingConfig,
        enums.ProblemType.PREDICTIVE_MODELING: RegressionTrainingConfig,
        enums.ProblemType.SENTENCE_BOUNDARY_DETECTION: SentenceBoundaryDetectionTrainingConfig,
        enums.ProblemType.SENTIMENT_DETECTION: SentimentDetectionTrainingConfig,
        enums.ProblemType.THEME_ANALYSIS: ThemeAnalysisTrainingConfig,
        enums.ProblemType.CUSTOM_ALGORITHM: CustomAlgorithmTrainingConfig,
        enums.ProblemType.OPTIMIZATION: OptimizationTrainingConfig
    }
