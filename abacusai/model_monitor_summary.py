from .return_class import AbstractApiClass


class ModelMonitorSummary(AbstractApiClass):
    """
        A summary of model monitor

        Args:
            client (ApiClient): An authenticated API Client instance
            monitorAccuracy (list): 
            modelDrift (list): 
            dataIntegrity (list): 
            biasViolations (list): 
    """

    def __init__(self, client, monitorAccuracy=None, modelDrift=None, dataIntegrity=None, biasViolations=None):
        super().__init__(client, None)
        self.monitor_accuracy = monitorAccuracy
        self.model_drift = modelDrift
        self.data_integrity = dataIntegrity
        self.bias_violations = biasViolations

    def __repr__(self):
        return f"ModelMonitorSummary(monitor_accuracy={repr(self.monitor_accuracy)},\n  model_drift={repr(self.model_drift)},\n  data_integrity={repr(self.data_integrity)},\n  bias_violations={repr(self.bias_violations)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'monitor_accuracy': self.monitor_accuracy, 'model_drift': self.model_drift, 'data_integrity': self.data_integrity, 'bias_violations': self.bias_violations}
