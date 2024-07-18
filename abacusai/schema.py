from .nested_feature_schema import NestedFeatureSchema
from .point_in_time_feature_info import PointInTimeFeatureInfo
from .return_class import AbstractApiClass


class Schema(AbstractApiClass):
    """
        A schema description for a feature

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The unique name of the feature.
            featureMapping (str): The mapping of the feature. The possible values will be based on the project's use-case. See the (Use Case Documentation)[https://api.abacus.ai/app/help/useCases] for more details.
            detectedFeatureMapping (str): Detected feature mapping for this feature
            featureType (str): The underlying data type of each feature:  CATEGORICAL,  CATEGORICAL_LIST,  NUMERICAL,  TIMESTAMP,  TEXT,  EMAIL,  LABEL_LIST,  ENTITY_LABEL_LIST,  PAGE_LABEL_LIST,  JSON,  OBJECT_REFERENCE,  MULTICATEGORICAL_LIST,  COORDINATE_LIST,  NUMERICAL_LIST,  TIMESTAMP_LIST,  ZIPCODE,  URL,  PAGE_INFOS,  PAGES_DOCUMENT,  TOKENS_DOCUMENT,  MESSAGE_LIST.
            detectedFeatureType (str): The detected feature type for this feature
            dataType (str): The underlying data type of each feature:  INTEGER,  FLOAT,  STRING,  DATE,  DATETIME,  BOOLEAN,  LIST,  STRUCT,  NULL,  BINARY.
            detectedDataType (str): The detected data type for this feature
            nestedFeatures (NestedFeatureSchema): List of features of nested feature
            pointInTimeInfo (PointInTimeFeatureInfo): Point in time information for this feature
    """

    def __init__(self, client, name=None, featureMapping=None, detectedFeatureMapping=None, featureType=None, detectedFeatureType=None, dataType=None, detectedDataType=None, nestedFeatures={}, pointInTimeInfo={}):
        super().__init__(client, None)
        self.name = name
        self.feature_mapping = featureMapping
        self.detected_feature_mapping = detectedFeatureMapping
        self.feature_type = featureType
        self.detected_feature_type = detectedFeatureType
        self.data_type = dataType
        self.detected_data_type = detectedDataType
        self.nested_features = client._build_class(
            NestedFeatureSchema, nestedFeatures)
        self.point_in_time_info = client._build_class(
            PointInTimeFeatureInfo, pointInTimeInfo)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'feature_mapping': repr(self.feature_mapping), f'detected_feature_mapping': repr(self.detected_feature_mapping), f'feature_type': repr(self.feature_type), f'detected_feature_type': repr(
            self.detected_feature_type), f'data_type': repr(self.data_type), f'detected_data_type': repr(self.detected_data_type), f'nested_features': repr(self.nested_features), f'point_in_time_info': repr(self.point_in_time_info)}
        class_name = "Schema"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'feature_mapping': self.feature_mapping, 'detected_feature_mapping': self.detected_feature_mapping, 'feature_type': self.feature_type, 'detected_feature_type': self.detected_feature_type,
                'data_type': self.data_type, 'detected_data_type': self.detected_data_type, 'nested_features': self._get_attribute_as_dict(self.nested_features), 'point_in_time_info': self._get_attribute_as_dict(self.point_in_time_info)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
