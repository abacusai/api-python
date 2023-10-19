from .return_class import AbstractApiClass


class FeatureGroupTemplateVariableOptions(AbstractApiClass):
    """
        Feature Group Template Variable Options

        Args:
            client (ApiClient): An authenticated API Client instance
            templateVariableOptions (list[dict]): List of values we can select for different template variables.
            userFeedback (list[str]): List of additional information regarding variable options for the user.
    """

    def __init__(self, client, templateVariableOptions=None, userFeedback=None):
        super().__init__(client, None)
        self.template_variable_options = templateVariableOptions
        self.user_feedback = userFeedback

    def __repr__(self):
        repr_dict = {f'template_variable_options': repr(
            self.template_variable_options), f'user_feedback': repr(self.user_feedback)}
        class_name = "FeatureGroupTemplateVariableOptions"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'template_variable_options': self.template_variable_options,
                'user_feedback': self.user_feedback}
        return {key: value for key, value in resp.items() if value is not None}
