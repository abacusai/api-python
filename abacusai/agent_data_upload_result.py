from .agent_data_document_info import AgentDataDocumentInfo
from .return_class import AbstractApiClass


class AgentDataUploadResult(AbstractApiClass):
    """
        Results of uploading data to agent.

        Args:
            client (ApiClient): An authenticated API Client instance
            docInfos (AgentDataDocumentInfo): A list of dict for information on the documents uploaded to agent.
    """

    def __init__(self, client, docInfos={}):
        super().__init__(client, None)
        self.doc_infos = client._build_class(AgentDataDocumentInfo, docInfos)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'doc_infos': repr(self.doc_infos)}
        class_name = "AgentDataUploadResult"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'doc_infos': self._get_attribute_as_dict(self.doc_infos)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
