from .return_class import AbstractApiClass


class AgentDataUploadResult(AbstractApiClass):
    """
        Results of uploading data to agent.

        Args:
            client (ApiClient): An authenticated API Client instance
            docIds (list[str]): A list of document IDs uploaded to agent.
    """

    def __init__(self, client, docIds=None):
        super().__init__(client, None)
        self.doc_ids = docIds

    def __repr__(self):
        repr_dict = {f'doc_ids': repr(self.doc_ids)}
        class_name = "AgentDataUploadResult"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'doc_ids': self.doc_ids}
        return {key: value for key, value in resp.items() if value is not None}
