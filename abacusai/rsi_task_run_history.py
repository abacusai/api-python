from .return_class import AbstractApiClass


class RsiTaskRunHistory(AbstractApiClass):
    """
        Run history of a recursively self-improving (RSI) daemon task

        Args:
            client (ApiClient): An authenticated API Client instance
            objective (str): The user's natural-language objective the task improves at.
            metricName (str): The name of the measured metric.
            direction (str): Whether the metric should 'increase' or 'decrease'.
            baselineValue (float): The metric value measured at task setup.
            minSample (int): Minimum sample size required to grade a run.
            noiseFloor (float): Absolute metric band treated as no change.
            lifecycle (str): The lifecycle of the daemon task.
            totalCount (int): Total number of runs with an iteration record.
            gradedCount (int): Number of runs whose iteration record has been graded (closed).
            limit (int): The page size used for the iterations list.
            offset (int): The offset used for the iterations list.
            iterations (list): One page of per-run records, newest first (iteration_index, status, value_before, value_after, delta, n_sample, verdict, change_summary, measured_at, failure_reason, run_record, created_at, run_lifecycle).
            trend (list): Compact full history for charting, oldest first (index, value, verdict; the in-flight run appears last with pending=true at its comparison baseline).
            sourceDeploymentConversationId (str): The conversation whose filesystem holds the RSI workdir (for fetching run records).
            workdir (str): The RSI working directory on that filesystem.
    """

    def __init__(self, client, objective=None, metricName=None, direction=None, baselineValue=None, minSample=None, noiseFloor=None, lifecycle=None, totalCount=None, gradedCount=None, limit=None, offset=None, iterations=None, trend=None, sourceDeploymentConversationId=None, workdir=None):
        super().__init__(client, None)
        self.objective = objective
        self.metric_name = metricName
        self.direction = direction
        self.baseline_value = baselineValue
        self.min_sample = minSample
        self.noise_floor = noiseFloor
        self.lifecycle = lifecycle
        self.total_count = totalCount
        self.graded_count = gradedCount
        self.limit = limit
        self.offset = offset
        self.iterations = iterations
        self.trend = trend
        self.source_deployment_conversation_id = sourceDeploymentConversationId
        self.workdir = workdir
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'objective': repr(self.objective), f'metric_name': repr(self.metric_name), f'direction': repr(self.direction), f'baseline_value': repr(self.baseline_value), f'min_sample': repr(self.min_sample), f'noise_floor': repr(self.noise_floor), f'lifecycle': repr(self.lifecycle), f'total_count': repr(
            self.total_count), f'graded_count': repr(self.graded_count), f'limit': repr(self.limit), f'offset': repr(self.offset), f'iterations': repr(self.iterations), f'trend': repr(self.trend), f'source_deployment_conversation_id': repr(self.source_deployment_conversation_id), f'workdir': repr(self.workdir)}
        class_name = "RsiTaskRunHistory"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'objective': self.objective, 'metric_name': self.metric_name, 'direction': self.direction, 'baseline_value': self.baseline_value, 'min_sample': self.min_sample, 'noise_floor': self.noise_floor, 'lifecycle': self.lifecycle, 'total_count': self.total_count,
                'graded_count': self.graded_count, 'limit': self.limit, 'offset': self.offset, 'iterations': self.iterations, 'trend': self.trend, 'source_deployment_conversation_id': self.source_deployment_conversation_id, 'workdir': self.workdir}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
