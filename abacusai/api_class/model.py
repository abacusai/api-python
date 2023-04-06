import dataclasses
from datetime import datetime
from typing import List

from . import enums
from .abstract import ApiClass, _ApiClassFactory


@dataclasses.dataclass
class TrainingConfig(ApiClass):
    _upper_snake_case_keys: bool = dataclasses.field(default=True, repr=False, init=False)
    _support_kwargs: bool = dataclasses.field(default=True, repr=False, init=False)

    kwargs: dict = dataclasses.field(default_factory=dict)
    problem_type: enums.ProblemType = dataclasses.field(default=None)


@dataclasses.dataclass
class ForecastingTrainingConfig(TrainingConfig):
    """
    Training config for the FORECASTING problem type
    Args:
        problem_type (ProblemType): FORECASTING
        prediction_length (int): How many timesteps in the future to predict.
        objective (ForecastingObjective): Ranking scheme used to select final best model.
        sort_objective (ForecastingObjective): Ranking scheme used to sort models on the metrics page.
        forecast_frequency (ForecastingFrequency): Forecast frequency.
        probability_quantiles (list[float]): Prediction quantiles.
        no_validation_set (bool): Do not generate validation set, test set will be used instead.
        force_prediction_length (int): Force length of test window to be the same as prediction length.
        filter_items (bool): Filter items with small history and volume.
        enable_cold_start (bool): Enable cold start forecasting by training/predicting for zero history items.
        enable_multiple_backtests (bool): Whether to enable multiple backtesting or not.
        total_backtesting_windows (int): Total backtesting windows to use for the training.
        backtest_window_step_size (int): Use this step size to shift backtesting windows for model training.
        full_data_retraining (bool): Train models separately with all the data.
        type_of_split (ForecastingDataSplitType): Type of data splitting into train/test.
        test_by_item (bool): Partition train/test data by item rather than time if true.
        test_start (datetime): Limit training data to dates before the given test start.
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
        prediction_offset (int): Offset for prediction.
        skip_local_scale_target (bool): Skip using per training/prediction window target scaling.
        timeseries_weight_column (str): If set, we use the values in this column from timeseries data to assign time dependent item weights during training and evaluation.
        item_attributes_weight_column (str): If set, we use the values in this column from item attributes data to assign weights to items during training and evaluation.
        use_timeseries_weights_in_objective (bool): If True, we include weights from column set as "TIMESERIES WEIGHT COLUMN" in objective functions.
        use_item_weights_in_objective (bool): If True, we include weights from column set as "ITEM ATTRIBUTES WEIGHT COLUMN" in objective functions.
        skip_timeseries_weight_scaling (bool): If True, we will avoid normalizing the weights.
        timeseries_loss_weight_column (str): Use value in this column to weight the loss while training.
        use_item_id (bool): Include a feature to indicate the item being forecast.
        use_all_item_totals (bool): Include as input total target across items.
        handle_zeros_as_missing (bool): If True, handle zero values in demand as missing data.
        datetime_holiday_calendars (list[HolidayCalendars]): Holiday calendars to augment training with.
        fill_missing_values (list[dict]): Strategy for filling in missing values.
        enable_clustering (bool): Enable clustering in forecasting.
        data_split_feature_group_table_name (str): Specify the table name of the feature group to export training data with the fold column.
        custom_loss_functions (list[str]): Registered custom losses available for selection.
        custom_metrics (list[str]): Registered custom metrics available for selection.
    """
    problem_type: enums.ProblemType = dataclasses.field(default=enums.ProblemType.FORECASTING, repr=False, init=False)
    prediction_length: int = dataclasses.field(default=None)
    objective: enums.ForecastingObjective = dataclasses.field(default=None)
    sort_objective: enums.ForecastingObjective = dataclasses.field(default=None)
    forecast_frequency: enums.ForecastingFrequency = dataclasses.field(default=None)
    probability_quantiles: List[float] = dataclasses.field(default=None)
    no_validation_set: bool = dataclasses.field(default=None)
    force_prediction_length: bool = dataclasses.field(default=None)
    filter_items: bool = dataclasses.field(default=None)
    enable_cold_start: bool = dataclasses.field(default=None)
    enable_multiple_backtests: bool = dataclasses.field(default=None)
    total_backtesting_windows: int = dataclasses.field(default=None)
    backtest_window_step_size: int = dataclasses.field(default=None)
    full_data_retraining: bool = dataclasses.field(default=None)
    # Data split params
    type_of_split: enums.ForecastingDataSplitType = dataclasses.field(default=enums.ForecastingDataSplitType.AUTO)
    test_by_item: bool = dataclasses.field(default=None)
    test_start: datetime = dataclasses.field(default=None)
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
    prediction_offset: int = dataclasses.field(default=None)
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
    handle_zeros_as_missing: bool = dataclasses.field(default=None)
    datetime_holiday_calendars: List[enums.HolidayCalendars] = dataclasses.field(default=None)
    fill_missing_values: List[dict] = dataclasses.field(default=None)
    enable_clustering: bool = dataclasses.field(default=None)
    # Others
    data_split_feature_group_table_name: str = dataclasses.field(default=None)
    custom_loss_functions: List[str] = dataclasses.field(default=None)
    custom_metrics: List[str] = dataclasses.field(default=None)


@dataclasses.dataclass
class _TrainingConfigFactory(_ApiClassFactory):
    config_abstract_class = TrainingConfig
    config_class_key = 'problem_type'
    config_class_map = {
        enums.ProblemType.FORECASTING: ForecastingTrainingConfig
    }
