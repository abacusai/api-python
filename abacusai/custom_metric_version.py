from .code_source import CodeSource
from .return_class import AbstractApiClass


class CustomMetricVersion(AbstractApiClass):
    """
        Custom metric version

        Args:
            client (ApiClient): An authenticated API Client instance
            customMetricVersion (str): Unique string identifier for the custom metric version.
            name (str): Name assigned to the custom metric.
            createdAt (str): ISO-8601 string indicating when the custom metric was created.
            customMetricFunctionName (str): The name of the function defined in the source code.
            codeSource (CodeSource): Information about the source code of the custom metric.
    """

    def __init__(self, client, customMetricVersion=None, name=None, createdAt=None, customMetricFunctionName=None, codeSource={}):
        super().__init__(client, customMetricVersion)
        self.custom_metric_version = customMetricVersion
        self.name = name
        self.created_at = createdAt
        self.custom_metric_function_name = customMetricFunctionName
        self.code_source = client._build_class(CodeSource, codeSource)

    def __repr__(self):
        repr_dict = {f'custom_metric_version': repr(self.custom_metric_version), f'name': repr(self.name), f'created_at': repr(
            self.created_at), f'custom_metric_function_name': repr(self.custom_metric_function_name), f'code_source': repr(self.code_source)}
        class_name = "CustomMetricVersion"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'custom_metric_version': self.custom_metric_version, 'name': self.name, 'created_at': self.created_at,
                'custom_metric_function_name': self.custom_metric_function_name, 'code_source': self._get_attribute_as_dict(self.code_source)}
        return {key: value for key, value in resp.items() if value is not None}

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            CustomMetricVersion: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Describes a given custom metric version

        Args:
            custom_metric_version (str): A unique string identifier for the custom metric version.

        Returns:
            CustomMetricVersion: An object describing the custom metric version.
        """
        return self.client.describe_custom_metric_version(self.custom_metric_version)
