from .return_class import AbstractApiClass


class AiBuildingTask(AbstractApiClass):
    """
        A task for Data Science Co-pilot to help build AI.

        Args:
            client (ApiClient): An authenticated API Client instance
            task (str): The task to be performed
            taskType (str): The type of task
    """

    def __init__(self, client, task=None, taskType=None):
        super().__init__(client, None)
        self.task = task
        self.task_type = taskType
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'task': repr(
            self.task), f'task_type': repr(self.task_type)}
        class_name = "AiBuildingTask"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'task': self.task, 'task_type': self.task_type}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
