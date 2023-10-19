from .return_class import AbstractApiClass


class DatasetColumn(AbstractApiClass):
    """
        A schema description for a column

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The unique name of the column.
            dataType (str): The underlying data type of each column. Refer to the [guide on data types](DATA_TYPES_URL) for more information.
            detectedDataType (str): The detected data type of the column.
            featureType (str): Feature type of the column.
            detectedFeatureType (str): The detected feature type of the column.
            originalName (str): The original name of the column.
            validDataTypes (list[str]): The valid data type options for this column.
            timeFormat (str): The detected time format of the column.
            timestampFrequency (str): The detected frequency of the timestamps in the dataset.
    """

    def __init__(self, client, name=None, dataType=None, detectedDataType=None, featureType=None, detectedFeatureType=None, originalName=None, validDataTypes=None, timeFormat=None, timestampFrequency=None):
        super().__init__(client, None)
        self.name = name
        self.data_type = dataType
        self.detected_data_type = detectedDataType
        self.feature_type = featureType
        self.detected_feature_type = detectedFeatureType
        self.original_name = originalName
        self.valid_data_types = validDataTypes
        self.time_format = timeFormat
        self.timestamp_frequency = timestampFrequency

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'data_type': repr(self.data_type), f'detected_data_type': repr(self.detected_data_type), f'feature_type': repr(self.feature_type), f'detected_feature_type': repr(
            self.detected_feature_type), f'original_name': repr(self.original_name), f'valid_data_types': repr(self.valid_data_types), f'time_format': repr(self.time_format), f'timestamp_frequency': repr(self.timestamp_frequency)}
        class_name = "DatasetColumn"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'data_type': self.data_type, 'detected_data_type': self.detected_data_type, 'feature_type': self.feature_type, 'detected_feature_type': self.detected_feature_type,
                'original_name': self.original_name, 'valid_data_types': self.valid_data_types, 'time_format': self.time_format, 'timestamp_frequency': self.timestamp_frequency}
        return {key: value for key, value in resp.items() if value is not None}
