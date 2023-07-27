from .return_class import AbstractApiClass


class AiBuildingTask(AbstractApiClass):
    """
        A task for AI Chat to help build AI.

        Args:
            client (ApiClient): An authenticated API Client instance
            task (str): The task to be performed
            taskType (str): The type of task
    """

    def __init__(self, client, task=None, taskType=None):
        super().__init__(client, None)
        self.task = task
        self.task_type = taskType

    def __repr__(self):
        return f"AiBuildingTask(task={repr(self.task)},\n  task_type={repr(self.task_type)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'task': self.task, 'task_type': self.task_type}
