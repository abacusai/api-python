from .return_class import AbstractApiClass


class AnnotationConfig(AbstractApiClass):
    """
        Annotation config for a feature group

        Args:
            client (ApiClient): An authenticated API Client instance
            featureAnnotationConfigs (list): List of feature annotation configs
            labels (list): List of labels
            statusFeature (str): Name of the feature that contains the status of the annotation (Optional)
            commentsFeature (str): Name of the feature that contains the comments for the annotation (Optional)
            metadataFeature (str): Name of the feature that contains the metadata for the annotation (Optional)
    """

    def __init__(self, client, featureAnnotationConfigs=None, labels=None, statusFeature=None, commentsFeature=None, metadataFeature=None):
        super().__init__(client, None)
        self.feature_annotation_configs = featureAnnotationConfigs
        self.labels = labels
        self.status_feature = statusFeature
        self.comments_feature = commentsFeature
        self.metadata_feature = metadataFeature

    def __repr__(self):
        return f"AnnotationConfig(feature_annotation_configs={repr(self.feature_annotation_configs)},\n  labels={repr(self.labels)},\n  status_feature={repr(self.status_feature)},\n  comments_feature={repr(self.comments_feature)},\n  metadata_feature={repr(self.metadata_feature)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'feature_annotation_configs': self.feature_annotation_configs, 'labels': self.labels, 'status_feature': self.status_feature, 'comments_feature': self.comments_feature, 'metadata_feature': self.metadata_feature}
