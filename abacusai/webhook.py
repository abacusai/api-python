from .return_class import AbstractApiClass


class Webhook(AbstractApiClass):
    """
        A Abacus.AI Webhook attached to an endpoint and event trigger for a given object.

        Args:
            client (ApiClient): An authenticated API Client instance
            webhookId (str): Unique identifier for this webhook.
            deploymentId (str): Identifier for the deployment this webhook is attached to.
            endpoint (str): The URI this webhook will send HTTP POST requests to.
            webhookEventType (str): The event that triggers the webhook action.
            payloadTemplate (str): Template for JSON Dictionary to be sent as the body of the POST request.
            createdAt (str): The date and time this webhook was created.
    """

    def __init__(self, client, webhookId=None, deploymentId=None, endpoint=None, webhookEventType=None, payloadTemplate=None, createdAt=None):
        super().__init__(client, webhookId)
        self.webhook_id = webhookId
        self.deployment_id = deploymentId
        self.endpoint = endpoint
        self.webhook_event_type = webhookEventType
        self.payload_template = payloadTemplate
        self.created_at = createdAt

    def __repr__(self):
        repr_dict = {f'webhook_id': repr(self.webhook_id), f'deployment_id': repr(self.deployment_id), f'endpoint': repr(self.endpoint), f'webhook_event_type': repr(
            self.webhook_event_type), f'payload_template': repr(self.payload_template), f'created_at': repr(self.created_at)}
        class_name = "Webhook"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'webhook_id': self.webhook_id, 'deployment_id': self.deployment_id, 'endpoint': self.endpoint,
                'webhook_event_type': self.webhook_event_type, 'payload_template': self.payload_template, 'created_at': self.created_at}
        return {key: value for key, value in resp.items() if value is not None}

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            Webhook: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Describe the webhook with a given ID.

        Args:
            webhook_id (str): Unique string identifier of the target webhook.

        Returns:
            Webhook: The webhook with the given ID.
        """
        return self.client.describe_webhook(self.webhook_id)

    def update(self, endpoint: str = None, webhook_event_type: str = None, payload_template: dict = None):
        """
        Update the webhook

        Args:
            endpoint (str): If provided, changes the webhook's endpoint.
            webhook_event_type (str): If provided, changes the event type.
            payload_template (dict): If provided, changes the payload template.
        """
        return self.client.update_webhook(self.webhook_id, endpoint, webhook_event_type, payload_template)

    def delete(self):
        """
        Delete the webhook

        Args:
            webhook_id (str): Unique identifier of the target webhook.
        """
        return self.client.delete_webhook(self.webhook_id)
