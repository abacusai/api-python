from .return_class import AbstractApiClass


class AnnotationsStatus(AbstractApiClass):
    """
        The status of annotations for a feature group

        Args:
            client (ApiClient): An authenticated API Client instance
            total (int): The total number of documents annotated
            done (int): The number of documents annotated
            inProgress (int): The number of documents currently being annotated
            todo (int): The number of documents that need to be annotated
            latestUpdatedAt (str): The latest time an annotation was updated (ISO-8601 format)
    """

    def __init__(self, client, total=None, done=None, inProgress=None, todo=None, latestUpdatedAt=None):
        super().__init__(client, None)
        self.total = total
        self.done = done
        self.in_progress = inProgress
        self.todo = todo
        self.latest_updated_at = latestUpdatedAt

    def __repr__(self):
        return f"AnnotationsStatus(total={repr(self.total)},\n  done={repr(self.done)},\n  in_progress={repr(self.in_progress)},\n  todo={repr(self.todo)},\n  latest_updated_at={repr(self.latest_updated_at)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'total': self.total, 'done': self.done, 'in_progress': self.in_progress, 'todo': self.todo, 'latest_updated_at': self.latest_updated_at}
