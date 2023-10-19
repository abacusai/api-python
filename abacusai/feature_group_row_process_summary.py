from .return_class import AbstractApiClass


class FeatureGroupRowProcessSummary(AbstractApiClass):
    """
        A summary of the feature group processes for a deployment.

        Args:
            client (ApiClient): An authenticated API Client instance
            totalProcesses (int): The total number of processes
            pendingProcesses (int): The number of pending processes
            processingProcesses (int): The number of processes currently processing
            completeProcesses (int): The number of complete processes
            failedProcesses (int): The number of failed processes
    """

    def __init__(self, client, totalProcesses=None, pendingProcesses=None, processingProcesses=None, completeProcesses=None, failedProcesses=None):
        super().__init__(client, None)
        self.total_processes = totalProcesses
        self.pending_processes = pendingProcesses
        self.processing_processes = processingProcesses
        self.complete_processes = completeProcesses
        self.failed_processes = failedProcesses

    def __repr__(self):
        repr_dict = {f'total_processes': repr(self.total_processes), f'pending_processes': repr(self.pending_processes), f'processing_processes': repr(
            self.processing_processes), f'complete_processes': repr(self.complete_processes), f'failed_processes': repr(self.failed_processes)}
        class_name = "FeatureGroupRowProcessSummary"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'total_processes': self.total_processes, 'pending_processes': self.pending_processes, 'processing_processes':
                self.processing_processes, 'complete_processes': self.complete_processes, 'failed_processes': self.failed_processes}
        return {key: value for key, value in resp.items() if value is not None}
