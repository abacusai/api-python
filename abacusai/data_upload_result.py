from .return_class import AbstractApiClass


class DataUploadResult(AbstractApiClass):
    """
        Results of uploading data to agent.

        Args:
            client (ApiClient): An authenticated API Client instance
            docInfos (list[agentdatadocumentinfo]): A list of dict for information on the documents uploaded to agent.
            maxCount (int): The maximum number of documents
    """

    def __init__(self, client, docInfos=None, maxCount=None):
        super().__init__(client, None)
        self.doc_infos = docInfos
        self.max_count = maxCount
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'doc_infos': repr(
            self.doc_infos), f'max_count': repr(self.max_count)}
        class_name = "DataUploadResult"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'doc_infos': self.doc_infos, 'max_count': self.max_count}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
