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
            mostRecentMonitorInfo (list): Most recent monitor information.
    """

    def __init__(self, client, summary=None, featureDrift=None, labelDrift=None, dataIntegrity=None, performance=None, alerts=None, mostRecentMonitorInfo=None):
        super().__init__(client, None)
        self.summary = summary
        self.feature_drift = featureDrift
        self.label_drift = labelDrift
        self.data_integrity = dataIntegrity
        self.performance = performance
        self.alerts = alerts
        self.most_recent_monitor_info = mostRecentMonitorInfo

    def __repr__(self):
        return f"ModelMonitorOrgSummary(summary={repr(self.summary)},\n  feature_drift={repr(self.feature_drift)},\n  label_drift={repr(self.label_drift)},\n  data_integrity={repr(self.data_integrity)},\n  performance={repr(self.performance)},\n  alerts={repr(self.alerts)},\n  most_recent_monitor_info={repr(self.most_recent_monitor_info)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'summary': self.summary, 'feature_drift': self.feature_drift, 'label_drift': self.label_drift, 'data_integrity': self.data_integrity, 'performance': self.performance, 'alerts': self.alerts, 'most_recent_monitor_info': self.most_recent_monitor_info}
