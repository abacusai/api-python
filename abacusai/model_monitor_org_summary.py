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

    def __repr__(self):
        return f"ModelMonitorOrgSummary(summary={repr(self.summary)},\n  feature_drift={repr(self.feature_drift)},\n  label_drift={repr(self.label_drift)},\n  data_integrity={repr(self.data_integrity)},\n  performance={repr(self.performance)},\n  alerts={repr(self.alerts)},\n  monitor_data={repr(self.monitor_data)},\n  total_starred_monitors={repr(self.total_starred_monitors)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'summary': self.summary, 'feature_drift': self.feature_drift, 'label_drift': self.label_drift, 'data_integrity': self.data_integrity, 'performance': self.performance, 'alerts': self.alerts, 'monitor_data': self.monitor_data, 'total_starred_monitors': self.total_starred_monitors}
