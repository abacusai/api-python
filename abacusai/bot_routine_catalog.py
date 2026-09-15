from .return_class import AbstractApiClass


class BotRoutineCatalog(AbstractApiClass):
    """
        The one-click routines a user can add to a personal agent

        Args:
            client (ApiClient): An authenticated API Client instance
            routines (list): The routines, in display order (id/title/description/avatar/color/categories/name/schedule/prompt, plus required_service and inputs where they apply).
            categories (list): The filter pills in display order (id/label); 'all' is the default view.
    """

    def __init__(self, client, routines=None, categories=None):
        super().__init__(client, None)
        self.routines = routines
        self.categories = categories
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'routines': repr(
            self.routines), f'categories': repr(self.categories)}
        class_name = "BotRoutineCatalog"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'routines': self.routines, 'categories': self.categories}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
