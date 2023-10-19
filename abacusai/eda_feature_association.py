from .return_class import AbstractApiClass


class EdaFeatureAssociation(AbstractApiClass):
    """
        Eda Feature Association between two features in the data.

        Args:
            client (ApiClient): An authenticated API Client instance
            data (dict): the data to display the feature association between two features
            isScatter (bool): A Boolean that represents if the data creates a scatter plot (for cases of numerical data vs numerical data)
            isBoxWhisker (bool): A Boolean that represents if the data creates a box whisker plot (For cases of categorical data vs numerical data and vice versa)
            xAxis (str): Name of the feature selected for feature association (reference_feature_name) for x axis on the plot
            yAxis (str): Name of the feature selected for feature association (test_feature_name) for y axis on the plot
            xAxisColumnValues (list): Name of all the categories within the x_axis feature (if it is a categorical data type)
            yAxisColumnValues (list): Name of all the categories within the y_axis feature (if it is a categorical data type)
            dataColumns (list): A list of columns listed in the data as keys
    """

    def __init__(self, client, data=None, isScatter=None, isBoxWhisker=None, xAxis=None, yAxis=None, xAxisColumnValues=None, yAxisColumnValues=None, dataColumns=None):
        super().__init__(client, None)
        self.data = data
        self.is_scatter = isScatter
        self.is_box_whisker = isBoxWhisker
        self.x_axis = xAxis
        self.y_axis = yAxis
        self.x_axis_column_values = xAxisColumnValues
        self.y_axis_column_values = yAxisColumnValues
        self.data_columns = dataColumns

    def __repr__(self):
        repr_dict = {f'data': repr(self.data), f'is_scatter': repr(self.is_scatter), f'is_box_whisker': repr(self.is_box_whisker), f'x_axis': repr(self.x_axis), f'y_axis': repr(
            self.y_axis), f'x_axis_column_values': repr(self.x_axis_column_values), f'y_axis_column_values': repr(self.y_axis_column_values), f'data_columns': repr(self.data_columns)}
        class_name = "EdaFeatureAssociation"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'data': self.data, 'is_scatter': self.is_scatter, 'is_box_whisker': self.is_box_whisker, 'x_axis': self.x_axis, 'y_axis': self.y_axis,
                'x_axis_column_values': self.x_axis_column_values, 'y_axis_column_values': self.y_axis_column_values, 'data_columns': self.data_columns}
        return {key: value for key, value in resp.items() if value is not None}
