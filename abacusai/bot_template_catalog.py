from .return_class import AbstractApiClass


class BotTemplateCatalog(AbstractApiClass):
    """
        The starter agent templates offered on the Personal Agents landing

        Args:
            client (ApiClient): An authenticated API Client instance
            templates (list): The templates, in display order (id/title/description/prompt/personality/avatar/color/categories/required_service_types, plus template, brand_icon, setup_checks, background_task, task_ideas and, when the org's disabled connectors rule it out, unavailable_reason where they apply). A setup check or background task needing mail/calendar carries required_service_type (MAIL / CALENDAR), required_service (the connector id it defaults to: the org's first allowed), available and connectors ([{service, scopes}]: the connectors the org allows, any one of which meets it, and the scopes the user's connection must hold). Chief of Staff is first.
            categories (list): The filter pills in display order (id/label); 'featured' is the default view.
            taskIdeas (list): Scheduled-task ideas the create-agent dialog offers (id/label/themes/name/prompt/schedule/cadence, plus routine_id, required_service_type (MAIL / CALENDAR) and uses (service types it reads when connected) where they apply). Templates without a setup checklist name theirs in task_ideas.
            taskIdeaThemes (list): The idea theme pills in display order (id/label); 'popular' is the default view.
            serviceConnectors (dict): Per service type (MAIL / CALENDAR), the connectors the org allows ([{service, scopes}]), so a typed task can be matched to them.
    """

    def __init__(self, client, templates=None, categories=None, taskIdeas=None, taskIdeaThemes=None, serviceConnectors=None):
        super().__init__(client, None)
        self.templates = templates
        self.categories = categories
        self.task_ideas = taskIdeas
        self.task_idea_themes = taskIdeaThemes
        self.service_connectors = serviceConnectors
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'templates': repr(self.templates), f'categories': repr(self.categories), f'task_ideas': repr(
            self.task_ideas), f'task_idea_themes': repr(self.task_idea_themes), f'service_connectors': repr(self.service_connectors)}
        class_name = "BotTemplateCatalog"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'templates': self.templates, 'categories': self.categories, 'task_ideas': self.task_ideas,
                'task_idea_themes': self.task_idea_themes, 'service_connectors': self.service_connectors}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
