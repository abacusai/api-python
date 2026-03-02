from .daemon_task_instance import DaemonTaskInstance
from .hosted_database import HostedDatabase
from .return_class import AbstractApiClass


class ChatllmTask(AbstractApiClass):
    """
        A chatllm task

        Args:
            client (ApiClient): An authenticated API Client instance
            chatllmTaskId (str): The id of the chatllm task.
            daemonTaskId (str): The id of the daemon task.
            taskType (str): The type of task ('chatllm' or 'daemon').
            name (str): The name of the chatllm task.
            instructions (str): The instructions of the chatllm task.
            description (str): The description of the chatllm task.
            lifecycle (str): The lifecycle of the chatllm task.
            scheduleInfo (dict): The schedule info of the chatllm task.
            externalApplicationId (str): The external application id associated with the chatllm task.
            deploymentConversationId (str): The deployment conversation id associated with the chatllm task.
            sourceDeploymentConversationId (str): The source deployment conversation id associated with the chatllm task.
            enableEmailAlerts (bool): Whether email alerts are enabled for the chatllm task.
            email (str): The email to send alerts to.
            numUnreadTaskInstances (int): The number of unread task instances for the chatllm task.
            computePointsUsed (int): The compute points used for the chatllm task.
            displayMarkdown (str): The display markdown for the chatllm task.
            requiresNewConversation (bool): Whether a new conversation is required for the chatllm task.
            executionMode (str): The execution mode of the chatllm task.
            taskDefinition (dict): The task definition (for web_service_trigger tasks).
            webAppHostname (str): The hostname of the web app associated with the daemon task.
            triggerType (str): The trigger type of the daemon task (scheduled or event_based).
            webhookUrl (str): The webhook URL for event-based daemon tasks.
            pushNotificationsEnabled (bool): Whether push notifications are enabled for the task.
            hostedDatabase (HostedDatabase): The hosted database for the daemon task.
            latestDaemonTaskInstance (DaemonTaskInstance): The latest task instance for daemon tasks.
    """

    def __init__(self, client, chatllmTaskId=None, daemonTaskId=None, taskType=None, name=None, instructions=None, description=None, lifecycle=None, scheduleInfo=None, externalApplicationId=None, deploymentConversationId=None, sourceDeploymentConversationId=None, enableEmailAlerts=None, email=None, numUnreadTaskInstances=None, computePointsUsed=None, displayMarkdown=None, requiresNewConversation=None, executionMode=None, taskDefinition=None, webAppHostname=None, triggerType=None, webhookUrl=None, pushNotificationsEnabled=None, hostedDatabase={}, latestDaemonTaskInstance={}):
        super().__init__(client, chatllmTaskId)
        self.chatllm_task_id = chatllmTaskId
        self.daemon_task_id = daemonTaskId
        self.task_type = taskType
        self.name = name
        self.instructions = instructions
        self.description = description
        self.lifecycle = lifecycle
        self.schedule_info = scheduleInfo
        self.external_application_id = externalApplicationId
        self.deployment_conversation_id = deploymentConversationId
        self.source_deployment_conversation_id = sourceDeploymentConversationId
        self.enable_email_alerts = enableEmailAlerts
        self.email = email
        self.num_unread_task_instances = numUnreadTaskInstances
        self.compute_points_used = computePointsUsed
        self.display_markdown = displayMarkdown
        self.requires_new_conversation = requiresNewConversation
        self.execution_mode = executionMode
        self.task_definition = taskDefinition
        self.web_app_hostname = webAppHostname
        self.trigger_type = triggerType
        self.webhook_url = webhookUrl
        self.push_notifications_enabled = pushNotificationsEnabled
        self.hosted_database = client._build_class(
            HostedDatabase, hostedDatabase)
        self.latest_daemon_task_instance = client._build_class(
            DaemonTaskInstance, latestDaemonTaskInstance)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'chatllm_task_id': repr(self.chatllm_task_id), f'daemon_task_id': repr(self.daemon_task_id), f'task_type': repr(self.task_type), f'name': repr(self.name), f'instructions': repr(self.instructions), f'description': repr(self.description), f'lifecycle': repr(self.lifecycle), f'schedule_info': repr(self.schedule_info), f'external_application_id': repr(self.external_application_id), f'deployment_conversation_id': repr(self.deployment_conversation_id), f'source_deployment_conversation_id': repr(self.source_deployment_conversation_id), f'enable_email_alerts': repr(self.enable_email_alerts), f'email': repr(self.email), f'num_unread_task_instances': repr(
            self.num_unread_task_instances), f'compute_points_used': repr(self.compute_points_used), f'display_markdown': repr(self.display_markdown), f'requires_new_conversation': repr(self.requires_new_conversation), f'execution_mode': repr(self.execution_mode), f'task_definition': repr(self.task_definition), f'web_app_hostname': repr(self.web_app_hostname), f'trigger_type': repr(self.trigger_type), f'webhook_url': repr(self.webhook_url), f'push_notifications_enabled': repr(self.push_notifications_enabled), f'hosted_database': repr(self.hosted_database), f'latest_daemon_task_instance': repr(self.latest_daemon_task_instance)}
        class_name = "ChatllmTask"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'chatllm_task_id': self.chatllm_task_id, 'daemon_task_id': self.daemon_task_id, 'task_type': self.task_type, 'name': self.name, 'instructions': self.instructions, 'description': self.description, 'lifecycle': self.lifecycle, 'schedule_info': self.schedule_info, 'external_application_id': self.external_application_id, 'deployment_conversation_id': self.deployment_conversation_id, 'source_deployment_conversation_id': self.source_deployment_conversation_id, 'enable_email_alerts': self.enable_email_alerts, 'email': self.email, 'num_unread_task_instances': self.num_unread_task_instances,
                'compute_points_used': self.compute_points_used, 'display_markdown': self.display_markdown, 'requires_new_conversation': self.requires_new_conversation, 'execution_mode': self.execution_mode, 'task_definition': self.task_definition, 'web_app_hostname': self.web_app_hostname, 'trigger_type': self.trigger_type, 'webhook_url': self.webhook_url, 'push_notifications_enabled': self.push_notifications_enabled, 'hosted_database': self._get_attribute_as_dict(self.hosted_database), 'latest_daemon_task_instance': self._get_attribute_as_dict(self.latest_daemon_task_instance)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
