import dataclasses

from . import enums
from .abstract import _ApiClassFactory
from .dataset import DatasetConfig


@dataclasses.dataclass
class StreamingConnectorDatasetConfig(DatasetConfig):
    """
    An abstract class for dataset configs specific to streaming connectors.

    Args:
        streaming_connector_type (StreamingConnectorType): The type of streaming connector
    """
    streaming_connector_type: enums.StreamingConnectorType = dataclasses.field(default=None, repr=False, init=False)

    @classmethod
    def _get_builder(cls):
        return _StreamingConnectorDatasetConfigFactory


@dataclasses.dataclass
class KafkaDatasetConfig(StreamingConnectorDatasetConfig):
    """
    Dataset config for Kafka Streaming Connector

    Args:
        topic (str): The kafka topic to consume
    """
    topic: str = dataclasses.field(default=None)

    def __post_init__(self):
        self.streaming_connector_type = enums.StreamingConnectorType.KAFKA


@dataclasses.dataclass
class _StreamingConnectorDatasetConfigFactory(_ApiClassFactory):
    config_abstract_class = StreamingConnectorDatasetConfig
    config_class_key = 'streaming_connector_type'
    config_class_map = {
        enums.StreamingConnectorType.KAFKA: KafkaDatasetConfig,
    }
