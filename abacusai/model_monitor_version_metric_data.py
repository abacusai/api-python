from .return_class import AbstractApiClass


class ModelMonitorVersionMetricData(AbstractApiClass):
    """
        Data for displaying model monitor version metric data

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name of the metric type
            algoName (str): The name of the algo used for the prediction metric
            featureGroupVersion (str): The prediction feature group used for analysis
            modelMonitor (str): The id of the model monitor
            modelMonitorVersion (str): The id of the model monitor version
            metricInfos (dict): Name and description for metrics
            metricNames (dict): Internal name to external name mapping
            metrics (dict): Metric name to metric data
            metricCharts (list): List of different metric charts
            otherMetrics (list): List of other metrics to optionally plot
            actualValuesSupportedForDrilldown (list): List of values support for drilldown
    """

    def __init__(self, client, name=None, algoName=None, featureGroupVersion=None, modelMonitor=None, modelMonitorVersion=None, metricInfos=None, metricNames=None, metrics=None, metricCharts=None, otherMetrics=None, actualValuesSupportedForDrilldown=None):
        super().__init__(client, None)
        self.name = name
        self.algo_name = algoName
        self.feature_group_version = featureGroupVersion
        self.model_monitor = modelMonitor
        self.model_monitor_version = modelMonitorVersion
        self.metric_infos = metricInfos
        self.metric_names = metricNames
        self.metrics = metrics
        self.metric_charts = metricCharts
        self.other_metrics = otherMetrics
        self.actual_values_supported_for_drilldown = actualValuesSupportedForDrilldown

    def __repr__(self):
        return f"ModelMonitorVersionMetricData(name={repr(self.name)},\n  algo_name={repr(self.algo_name)},\n  feature_group_version={repr(self.feature_group_version)},\n  model_monitor={repr(self.model_monitor)},\n  model_monitor_version={repr(self.model_monitor_version)},\n  metric_infos={repr(self.metric_infos)},\n  metric_names={repr(self.metric_names)},\n  metrics={repr(self.metrics)},\n  metric_charts={repr(self.metric_charts)},\n  other_metrics={repr(self.other_metrics)},\n  actual_values_supported_for_drilldown={repr(self.actual_values_supported_for_drilldown)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'algo_name': self.algo_name, 'feature_group_version': self.feature_group_version, 'model_monitor': self.model_monitor, 'model_monitor_version': self.model_monitor_version, 'metric_infos': self.metric_infos, 'metric_names': self.metric_names, 'metrics': self.metrics, 'metric_charts': self.metric_charts, 'other_metrics': self.other_metrics, 'actual_values_supported_for_drilldown': self.actual_values_supported_for_drilldown}
