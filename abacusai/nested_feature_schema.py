from .point_in_time_feature_info import PointInTimeFeatureInfo
from .return_class import AbstractApiClass


class NestedFeatureSchema(AbstractApiClass):
    """
        A schema description for a nested feature

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The unique name of the column
            featureType (str): Feature Type of the Feature
            featureMapping (str): The Feature Mapping of the feature
            dataType (str): Data Type of the Feature
            detectedFeatureType (str): The detected feature type for this feature
            sourceTable (str): The source table of the column
            pointInTimeInfo (PointInTimeFeatureInfo): Point in time information for this feature
    """

    def __init__(self, client, name=None, featureType=None, featureMapping=None, dataType=None, detectedFeatureType=None, sourceTable=None, pointInTimeInfo={}):
        super().__init__(client, None)
        self.name = name
        self.feature_type = featureType
        self.feature_mapping = featureMapping
        self.data_type = dataType
        self.detected_feature_type = detectedFeatureType
        self.source_table = sourceTable
        self.point_in_time_info = client._build_class(
            PointInTimeFeatureInfo, pointInTimeInfo)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'feature_type': repr(self.feature_type), f'feature_mapping': repr(self.feature_mapping), f'data_type': repr(
            self.data_type), f'detected_feature_type': repr(self.detected_feature_type), f'source_table': repr(self.source_table), f'point_in_time_info': repr(self.point_in_time_info)}
        class_name = "NestedFeatureSchema"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'feature_type': self.feature_type, 'feature_mapping': self.feature_mapping, 'data_type': self.data_type,
                'detected_feature_type': self.detected_feature_type, 'source_table': self.source_table, 'point_in_time_info': self._get_attribute_as_dict(self.point_in_time_info)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
