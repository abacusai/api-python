from .organization_chatllm_project import OrganizationChatllmProject
from .return_class import AbstractApiClass


class OrganizationChatllmProjectPage(AbstractApiClass):
    """
        One page of the organization's ChatLLM projects (listOrganizationChatllmProjects).

        Args:
            client (ApiClient): An authenticated API Client instance
            totalCount (int): How many projects match in all, so a caller knows when `offset` has reached the end.
            items (OrganizationChatllmProject): The projects on this page, newest first.
    """

    def __init__(self, client, totalCount=None, items={}):
        super().__init__(client, None)
        self.total_count = totalCount
        self.items = client._build_class(OrganizationChatllmProject, items)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'total_count': repr(
            self.total_count), f'items': repr(self.items)}
        class_name = "OrganizationChatllmProjectPage"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'total_count': self.total_count,
                'items': self._get_attribute_as_dict(self.items)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
