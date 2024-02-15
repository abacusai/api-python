from .return_class import AbstractApiClass


class ModelMonitorOrgSummary(AbstractApiClass):
    """
        A summary of an organization's model monitors

        Args:
            client (ApiClient): An authenticated API Client instance
            summary (dict): Count of monitors, count of versions, count of total rows of prediction data, count of failed versions.
            featureDrift (dict): Percentage of monitors with and without KL divergence > 2.
            labelDrift (dict): Histogram of label drift across versions.
            dataIntegrity (dict): Counts of violations.
            performance (dict): Model accuracy information.
            alerts (dict): Count of alerts that are raised.
            monitorData (dict): Information about monitors used in the summary for each time period.
            totalStarredMonitors (int): Total number of starred monitors.
    """

    def __init__(self, client, summary=None, featureDrift=None, labelDrift=None, dataIntegrity=None, performance=None, alerts=None, monitorData=None, totalStarredMonitors=None):
        super().__init__(client, None)
        self.summary = summary
        self.feature_drift = featureDrift
        self.label_drift = labelDrift
        self.data_integrity = dataIntegrity
        self.performance = performance
        self.alerts = alerts
        self.monitor_data = monitorData
        self.total_starred_monitors = totalStarredMonitors
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'summary': repr(self.summary), f'feature_drift': repr(self.feature_drift), f'label_drift': repr(self.label_drift), f'data_integrity': repr(self.data_integrity), f'performance': repr(
            self.performance), f'alerts': repr(self.alerts), f'monitor_data': repr(self.monitor_data), f'total_starred_monitors': repr(self.total_starred_monitors)}
        class_name = "ModelMonitorOrgSummary"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'summary': self.summary, 'feature_drift': self.feature_drift, 'label_drift': self.label_drift, 'data_integrity': self.data_integrity,
                'performance': self.performance, 'alerts': self.alerts, 'monitor_data': self.monitor_data, 'total_starred_monitors': self.total_starred_monitors}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
