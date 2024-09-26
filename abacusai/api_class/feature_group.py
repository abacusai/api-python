import dataclasses
from typing import List

from . import enums
from .abstract import ApiClass, _ApiClassFactory
from .dataset import DocumentProcessingConfig


@dataclasses.dataclass
class SamplingConfig(ApiClass):
    """
    An abstract class for the sampling config of a feature group
    """
    sampling_method: enums.SamplingMethodType = dataclasses.field(default=None, repr=False, init=False)

    @classmethod
    def _get_builder(cls):
        return _SamplingConfigFactory

    def __post_init__(self):
        if self.__class__ == SamplingConfig:
            raise TypeError('Cannot instantiate abstract SamplingConfig class.')


@dataclasses.dataclass
class NSamplingConfig(SamplingConfig):
    """
    The number of distinct values of the key columns to include in the sample, or number of rows if key columns not specified.

    Args:
        sample_count (int): The number of rows to include in the sample
        key_columns (List[str]): The feature(s) to use as the key(s) when sampling
    """
    sample_count: int
    key_columns: List[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        self.sampling_method = enums.SamplingMethodType.N_SAMPLING


@dataclasses.dataclass
class PercentSamplingConfig(SamplingConfig):
    """
    The fraction of distinct values of the feature group to include in the sample.

    Args:
        sample_percent (float): The percentage of the rows to sample
        key_columns (List[str]): The feature(s) to use as the key(s) when sampling
    """
    sample_percent: float
    key_columns: List[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        self.sampling_method = enums.SamplingMethodType.PERCENT_SAMPLING


@dataclasses.dataclass
class _SamplingConfigFactory(_ApiClassFactory):
    config_class_key = 'sampling_method'
    config_abstract_class = SamplingConfig
    config_class_map = {
        enums.SamplingMethodType.N_SAMPLING: NSamplingConfig,
        enums.SamplingMethodType.PERCENT_SAMPLING: PercentSamplingConfig,
    }


@dataclasses.dataclass
class MergeConfig(ApiClass):
    """
    An abstract class for the merge config of a feature group
    """
    merge_mode: enums.MergeMode = dataclasses.field(default=None, repr=False, init=False)

    @classmethod
    def _get_builder(self):
        return _MergeConfigFactory

    def __post_init__(self):
        if self.__class__ == MergeConfig:
            raise TypeError('Cannot instantiate abstract MergeConfig class.')


@dataclasses.dataclass
class LastNMergeConfig(MergeConfig):
    """
    Merge LAST N chunks/versions of an incremental dataset.

    Args:
        num_versions (int): The number of versions to merge. num_versions == 0 means merge all versions.
        include_version_timestamp_column (bool): If set, include a column with the creation timestamp of source FG versions.
    """
    num_versions: int
    include_version_timestamp_column: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.merge_mode = enums.MergeMode.LAST_N


@dataclasses.dataclass
class TimeWindowMergeConfig(MergeConfig):
    """
    Merge rows within a given timewindow of the most recent timestamp

    Args:
        feature_name (str): Time based column to index on
        time_window_size_ms (int): Range of merged rows will be [MAX_TIME - time_window_size_ms, MAX_TIME]
        include_version_timestamp_column (bool): If set, include a column with the creation timestamp of source FG versions.
    """
    feature_name: str
    time_window_size_ms: int
    include_version_timestamp_column: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.merge_mode = enums.MergeMode.TIME_WINDOW


@dataclasses.dataclass
class _MergeConfigFactory(_ApiClassFactory):
    config_class_key = 'merge_mode'
    config_abstract_class = MergeConfig
    config_class_map = {
        enums.MergeMode.LAST_N: LastNMergeConfig,
        enums.MergeMode.TIME_WINDOW: TimeWindowMergeConfig,
    }


@dataclasses.dataclass
class OperatorConfig(ApiClass):
    """Configuration for a template Feature Group Operation"""
    operator_type: enums.OperatorType = dataclasses.field(default=None, repr=False, init=False)

    @classmethod
    def _get_builder(cls):
        return _OperatorConfigFactory

    def __post_init__(self):
        if self.__class__ == OperatorConfig:
            raise TypeError('Cannot instantiate abstract OperatorConfig class.')


@dataclasses.dataclass
class UnpivotConfig(OperatorConfig):
    """ Unpivot Columns in a FeatureGroup.

    Args:
        columns (List[str]): Which columns to unpivot.
        index_column (str): Name of new column containing the unpivoted column names as its values
        value_column (str): Name of new column containing the row values that were unpivoted.
        exclude (bool): If True, the unpivoted columns are all the columns EXCEPT the ones in the columns argument. Default is False.
    """

    columns: List[str] = dataclasses.field(default=None)
    index_column: str = dataclasses.field(default=None)
    value_column: str = dataclasses.field(default=None)
    exclude: bool = dataclasses.field(default=None)

    def __post_init__(self):
        self.operator_type = enums.OperatorType.UNPIVOT


@dataclasses.dataclass
class MarkdownConfig(OperatorConfig):
    """ Transform a input column to a markdown column.

    Args:
        input_column (str): Name of input column to transform.
        output_column (str): Name of output column to store transformed data.
        input_column_type (MarkdownOperatorInputType): Type of input column to transform.
    """
    input_column: str = dataclasses.field(default=None)
    output_column: str = dataclasses.field(default=None)
    input_column_type: enums.MarkdownOperatorInputType = dataclasses.field(default=None)

    def __post_init__(self):
        self.operator_type = enums.OperatorType.MARKDOWN


@dataclasses.dataclass
class CrawlerTransformConfig(OperatorConfig):
    """ Transform a input column of urls to html text

    Args:
        input_column (str): Name of input column to transform.
        output_column (str): Name of output column to store transformed data.
        depth_column (str): Increasing depth explores more links, capturing more content
        disable_host_restriction (bool): If True, will not restrict crawling to the same host.
        honour_website_rules (bool): If True, will respect robots.txt rules.
        user_agent (str): If provided, will use this user agent instead of randomly selecting one.
    """
    input_column: str = dataclasses.field(default=None)
    output_column: str = dataclasses.field(default=None)
    depth_column: str = dataclasses.field(default=None)
    input_column_type: str = dataclasses.field(default=None, metadata={'deprecated': True})
    crawl_depth: int = dataclasses.field(default=None, metadata={'deprecated': True})
    disable_host_restriction: bool = dataclasses.field(default=None)
    honour_website_rules: bool = dataclasses.field(default=None)
    user_agent: str = dataclasses.field(default=None)

    def __post_init__(self):
        self.operator_type = enums.OperatorType.CRAWLER


@dataclasses.dataclass
class ExtractDocumentDataConfig(OperatorConfig):
    """ Extracts data from documents.

    Args:
        doc_id_column (str): Name of input document ID column.
        document_column (str): Name of the input document column which contains the page infos. This column will be transformed to include the document processing config in the output feature group.
        document_processing_config (DocumentProcessingConfig): Document processing configuration.
    """
    doc_id_column: str = dataclasses.field(default=None)
    document_column: str = dataclasses.field(default=None)
    document_processing_config: DocumentProcessingConfig = dataclasses.field(default=None)

    def __post_init__(self):
        self.operator_type = enums.OperatorType.EXTRACT_DOCUMENT_DATA


# TODO: create nested dict object so this does not need to be in sync with UI.
@dataclasses.dataclass
class DataGenerationConfig(OperatorConfig):
    """ Generate synthetic data using a model for finetuning an LLM.

    Args:
        prompt_col (str): Name of the input prompt column.
        completion_col (str): Name of the output completion column.
        description_col (str): Name of the description column.
        id_col (str): Name of the identifier column.
        generation_instructions (str): Instructions for the data generation model.
        temperature (float): Sampling temperature for the model.
        fewshot_examples (int): Number of fewshot examples used to prompt the model.
        concurrency (int): Number of concurrent processes.
        examples_per_target (int): Number of examples per target.
        subset_size (Optional[int]): Size of the subset to use for generation.
        verify_response (bool): Whether to verify the response.
        token_budget (int): Token budget for generation.
        oversample (bool): Whether to oversample the data.
        documentation_char_limit (int): Character limit for documentation.
        frequency_penalty (float): Penalty for frequency of token appearance.
        model (str): Model to use for data generation.
        seed (Optional[int]): Seed for random number generation.
    """
    # required
    prompt_col: str = dataclasses.field(default=None)
    completion_col: str = dataclasses.field(default=None)
    description_col: str = dataclasses.field(default=None)
    id_col: str = dataclasses.field(default=None)
    generation_instructions: str = dataclasses.field(default=None)

    # optional
    temperature: float = dataclasses.field(default=None)
    fewshot_examples: int = dataclasses.field(default=None)
    concurrency: int = dataclasses.field(default=None)
    examples_per_target: int = dataclasses.field(default=None)
    subset_size: int = dataclasses.field(default=None)
    verify_response: bool = dataclasses.field(default=None)
    token_budget: int = dataclasses.field(default=None)
    oversample: bool = dataclasses.field(default=None)
    documentation_char_limit: int = dataclasses.field(default=None)
    frequency_penalty: float = dataclasses.field(default=None)
    model: str = dataclasses.field(default=None)
    seed: int = dataclasses.field(default=None)

    def __post_init__(self):
        self.operator_type = enums.OperatorType.DATA_GENERATION


@dataclasses.dataclass
class UnionTransformConfig(OperatorConfig):
    """Takes Union of current feature group with 1 or more selected feature groups of same type.

    Args:
        feature_group_ids (List[str]): List of feature group IDs to union with source FG.
        drop_non_intersecting_columns (bool): If true, will drop columns that are not present in all feature groups. If false fills missing columns with nulls.
    """
    feature_group_ids: List[str] = dataclasses.field(default=None)
    drop_non_intersecting_columns: bool = dataclasses.field(default=False)

    def __post_init__(self):
        self.operator_type = enums.OperatorType.UNION


@dataclasses.dataclass
class _OperatorConfigFactory(_ApiClassFactory):
    """A class to select and return the the correct type of Operator Config based on a serialized OperatorConfig instance. """
    config_abstract_class = OperatorConfig
    config_class_key = 'operator_type'
    config_class_map = {
        enums.OperatorType.UNPIVOT: UnpivotConfig,
        enums.OperatorType.MARKDOWN: MarkdownConfig,
        enums.OperatorType.CRAWLER: CrawlerTransformConfig,
        enums.OperatorType.EXTRACT_DOCUMENT_DATA: ExtractDocumentDataConfig,
        enums.OperatorType.DATA_GENERATION: DataGenerationConfig,
        enums.OperatorType.UNION: UnionTransformConfig,
    }
