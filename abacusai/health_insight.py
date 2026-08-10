from .return_class import AbstractApiClass


class HealthInsight(AbstractApiClass):
    """
        A generated Health insight card (grounded in the user's connected wearable data).

        Args:
            client (ApiClient): An authenticated API Client instance
            healthInsightId (id): The ID of the insight card.
            category (str): heart | sleep | activity | summary.
            timeRange (str): week | month | 3month.
            title (str): Short headline.
            body (str): 1-2 sentence insight body.
            sourceProvider (str): Provider the insight is attributed to (oura, fitbit, ...).
            followUpPrompt (str): Suggested first-person question for the "Ask ->" chat follow-up.
            status (str): current | stale.
            generatedAt (str): When the card was generated.
    """

    def __init__(self, client, healthInsightId=None, category=None, timeRange=None, title=None, body=None, sourceProvider=None, followUpPrompt=None, status=None, generatedAt=None):
        super().__init__(client, healthInsightId)
        self.health_insight_id = healthInsightId
        self.category = category
        self.time_range = timeRange
        self.title = title
        self.body = body
        self.source_provider = sourceProvider
        self.follow_up_prompt = followUpPrompt
        self.status = status
        self.generated_at = generatedAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'health_insight_id': repr(self.health_insight_id), f'category': repr(self.category), f'time_range': repr(self.time_range), f'title': repr(self.title), f'body': repr(
            self.body), f'source_provider': repr(self.source_provider), f'follow_up_prompt': repr(self.follow_up_prompt), f'status': repr(self.status), f'generated_at': repr(self.generated_at)}
        class_name = "HealthInsight"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'health_insight_id': self.health_insight_id, 'category': self.category, 'time_range': self.time_range, 'title': self.title, 'body': self.body,
                'source_provider': self.source_provider, 'follow_up_prompt': self.follow_up_prompt, 'status': self.status, 'generated_at': self.generated_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
