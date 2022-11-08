from .return_class import AbstractApiClass


class ModelMonitorSummary(AbstractApiClass):
    """
        A summary of model monitor

        Args:
            client (ApiClient): An authenticated API Client instance
            modelAccuracy (list): A list of model accuracy objects including accuracy and monitor version information.
            modelDrift (list): A list of model drift objects including label and prediction drifts and monitor version information.
            dataIntegrity (list): A list of data integrity objects including counts of violations and monitor version information.
            biasViolations (list): A list of bias objects including bias counts and monitor version information.
            alerts (list): A list of alerts by type for each model monitor instance
    """

    def __init__(self, client, modelAccuracy=None, modelDrift=None, dataIntegrity=None, biasViolations=None, alerts=None):
        super().__init__(client, None)
        self.model_accuracy = modelAccuracy
        self.model_drift = modelDrift
        self.data_integrity = dataIntegrity
        self.bias_violations = biasViolations
        self.alerts = alerts

    def __repr__(self):
        return f"ModelMonitorSummary(model_accuracy={repr(self.model_accuracy)},\n  model_drift={repr(self.model_drift)},\n  data_integrity={repr(self.data_integrity)},\n  bias_violations={repr(self.bias_violations)},\n  alerts={repr(self.alerts)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'model_accuracy': self.model_accuracy, 'model_drift': self.model_drift, 'data_integrity': self.data_integrity, 'bias_violations': self.bias_violations, 'alerts': self.alerts}
