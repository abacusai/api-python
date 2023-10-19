from .annotation_config import AnnotationConfig
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
            isMaterializationNeeded (bool): Whether feature group needs to be materialized before using for annotations
            latestMaterializedAnnotationConfig (AnnotationConfig): The annotation config corresponding to the latest materialized feature group
    """

    def __init__(self, client, total=None, done=None, inProgress=None, todo=None, latestUpdatedAt=None, isMaterializationNeeded=None, latestMaterializedAnnotationConfig={}):
        super().__init__(client, None)
        self.total = total
        self.done = done
        self.in_progress = inProgress
        self.todo = todo
        self.latest_updated_at = latestUpdatedAt
        self.is_materialization_needed = isMaterializationNeeded
        self.latest_materialized_annotation_config = client._build_class(
            AnnotationConfig, latestMaterializedAnnotationConfig)

    def __repr__(self):
        repr_dict = {f'total': repr(self.total), f'done': repr(self.done), f'in_progress': repr(self.in_progress), f'todo': repr(self.todo), f'latest_updated_at': repr(
            self.latest_updated_at), f'is_materialization_needed': repr(self.is_materialization_needed), f'latest_materialized_annotation_config': repr(self.latest_materialized_annotation_config)}
        class_name = "AnnotationsStatus"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'total': self.total, 'done': self.done, 'in_progress': self.in_progress, 'todo': self.todo, 'latest_updated_at': self.latest_updated_at,
                'is_materialization_needed': self.is_materialization_needed, 'latest_materialized_annotation_config': self._get_attribute_as_dict(self.latest_materialized_annotation_config)}
        return {key: value for key, value in resp.items() if value is not None}
