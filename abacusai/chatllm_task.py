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
            sourceDeploymentConversationName (str): The name of the source deployment conversation, for linking back to it.
            latestSourceDeploymentConversationId (str): The most recent conversation in the source conversation's thread family (excluding task-run conversations).
            enableEmailAlerts (bool): Whether email alerts are enabled for the chatllm task.
            email (str): The email to send alerts to.
            numUnreadTaskInstances (int): The number of unread task instances for the chatllm task.
            computePointsUsed (int): The compute points used for the chatllm task.
            displayMarkdown (str): The display markdown for the chatllm task.
            requiresNewConversation (bool): Whether a new conversation is required for the chatllm task.
            executionMode (str): The execution mode of the chatllm task.
            offerModelSwitch (bool): Whether to offer switching to a cheaper model.
            taskDefinition (dict): The task definition (for web_service_trigger tasks).
            webAppHostname (str): The hostname of the web app associated with the daemon task.
            triggerType (str): The trigger type of the daemon task (scheduled or event_based).
            nextRunSkipped (bool): Whether the next scheduled run was stopped (skipped) by the user.
            skippedRunTime (int): The unix timestamp of the run being skipped, when next_run_skipped is true.
            webhookUrl (str): The webhook URL for event-based daemon tasks.
            pushNotificationsEnabled (bool): Whether push notifications are enabled for the task.
            skills (list): Names of agent skills pinned to this daemon task.
            isRsi (bool): Whether the daemon task is a recursively self-improving (RSI) task.
            isSupercomputer (bool): Whether the daemon task runs on the user's SuperComputer VM (pinned via its conversation).
            personalAgentComputerId (str): The id of the SuperComputer VM the task is pinned to, when is_supercomputer is true.
            currentEffort (str): The worker effort the task currently runs on (auto/simple/medium/high/xhigh/max).
            effortLadder (list): The pickable effort rungs available for this task.
            isOwner (bool): Whether the requesting user owns the task.
            isOrgShared (bool): Whether the task is shared org-wide.
            isDirectlyShared (bool): Whether the task was shared with the requesting user or their groups directly; only set for non-owners.
            isProjectShared (bool): Whether the task is visible because its source conversation belongs to a project the requesting user is a member of; only set for non-owners.
            sharedPermission (str): The permission the task was shared with ('VIEW' or 'EDIT'); only set for non-owners.
            ownerEmail (str): The task owner's email; only set for non-owners.
            resumeRequestId (str): The streaming request id for a reply/follow-up just dispatched via answer_daemon_task_input, for the UI to attach the live stream immediately.
            hostedDatabase (HostedDatabase): The hosted database for the daemon task.
            latestDaemonTaskInstance (DaemonTaskInstance): The latest task instance for daemon tasks.
    """

    def __init__(self, client, chatllmTaskId=None, daemonTaskId=None, taskType=None, name=None, instructions=None, description=None, lifecycle=None, scheduleInfo=None, externalApplicationId=None, deploymentConversationId=None, sourceDeploymentConversationId=None, sourceDeploymentConversationName=None, latestSourceDeploymentConversationId=None, enableEmailAlerts=None, email=None, numUnreadTaskInstances=None, computePointsUsed=None, displayMarkdown=None, requiresNewConversation=None, executionMode=None, offerModelSwitch=None, taskDefinition=None, webAppHostname=None, triggerType=None, nextRunSkipped=None, skippedRunTime=None, webhookUrl=None, pushNotificationsEnabled=None, skills=None, isRsi=None, isSupercomputer=None, personalAgentComputerId=None, currentEffort=None, effortLadder=None, isOwner=None, isOrgShared=None, isDirectlyShared=None, isProjectShared=None, sharedPermission=None, ownerEmail=None, resumeRequestId=None, hostedDatabase={}, latestDaemonTaskInstance={}):
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
        self.source_deployment_conversation_name = sourceDeploymentConversationName
        self.latest_source_deployment_conversation_id = latestSourceDeploymentConversationId
        self.enable_email_alerts = enableEmailAlerts
        self.email = email
        self.num_unread_task_instances = numUnreadTaskInstances
        self.compute_points_used = computePointsUsed
        self.display_markdown = displayMarkdown
        self.requires_new_conversation = requiresNewConversation
        self.execution_mode = executionMode
        self.offer_model_switch = offerModelSwitch
        self.task_definition = taskDefinition
        self.web_app_hostname = webAppHostname
        self.trigger_type = triggerType
        self.next_run_skipped = nextRunSkipped
        self.skipped_run_time = skippedRunTime
        self.webhook_url = webhookUrl
        self.push_notifications_enabled = pushNotificationsEnabled
        self.skills = skills
        self.is_rsi = isRsi
        self.is_supercomputer = isSupercomputer
        self.personal_agent_computer_id = personalAgentComputerId
        self.current_effort = currentEffort
        self.effort_ladder = effortLadder
        self.is_owner = isOwner
        self.is_org_shared = isOrgShared
        self.is_directly_shared = isDirectlyShared
        self.is_project_shared = isProjectShared
        self.shared_permission = sharedPermission
        self.owner_email = ownerEmail
        self.resume_request_id = resumeRequestId
        self.hosted_database = client._build_class(
            HostedDatabase, hostedDatabase)
        self.latest_daemon_task_instance = client._build_class(
            DaemonTaskInstance, latestDaemonTaskInstance)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'chatllm_task_id': repr(self.chatllm_task_id), f'daemon_task_id': repr(self.daemon_task_id), f'task_type': repr(self.task_type), f'name': repr(self.name), f'instructions': repr(self.instructions), f'description': repr(self.description), f'lifecycle': repr(self.lifecycle), f'schedule_info': repr(self.schedule_info), f'external_application_id': repr(self.external_application_id), f'deployment_conversation_id': repr(self.deployment_conversation_id), f'source_deployment_conversation_id': repr(self.source_deployment_conversation_id), f'source_deployment_conversation_name': repr(self.source_deployment_conversation_name), f'latest_source_deployment_conversation_id': repr(self.latest_source_deployment_conversation_id), f'enable_email_alerts': repr(self.enable_email_alerts), f'email': repr(self.email), f'num_unread_task_instances': repr(self.num_unread_task_instances), f'compute_points_used': repr(self.compute_points_used), f'display_markdown': repr(self.display_markdown), f'requires_new_conversation': repr(self.requires_new_conversation), f'execution_mode': repr(self.execution_mode), f'offer_model_switch': repr(
            self.offer_model_switch), f'task_definition': repr(self.task_definition), f'web_app_hostname': repr(self.web_app_hostname), f'trigger_type': repr(self.trigger_type), f'next_run_skipped': repr(self.next_run_skipped), f'skipped_run_time': repr(self.skipped_run_time), f'webhook_url': repr(self.webhook_url), f'push_notifications_enabled': repr(self.push_notifications_enabled), f'skills': repr(self.skills), f'is_rsi': repr(self.is_rsi), f'is_supercomputer': repr(self.is_supercomputer), f'personal_agent_computer_id': repr(self.personal_agent_computer_id), f'current_effort': repr(self.current_effort), f'effort_ladder': repr(self.effort_ladder), f'is_owner': repr(self.is_owner), f'is_org_shared': repr(self.is_org_shared), f'is_directly_shared': repr(self.is_directly_shared), f'is_project_shared': repr(self.is_project_shared), f'shared_permission': repr(self.shared_permission), f'owner_email': repr(self.owner_email), f'resume_request_id': repr(self.resume_request_id), f'hosted_database': repr(self.hosted_database), f'latest_daemon_task_instance': repr(self.latest_daemon_task_instance)}
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
        resp = {'chatllm_task_id': self.chatllm_task_id, 'daemon_task_id': self.daemon_task_id, 'task_type': self.task_type, 'name': self.name, 'instructions': self.instructions, 'description': self.description, 'lifecycle': self.lifecycle, 'schedule_info': self.schedule_info, 'external_application_id': self.external_application_id, 'deployment_conversation_id': self.deployment_conversation_id, 'source_deployment_conversation_id': self.source_deployment_conversation_id, 'source_deployment_conversation_name': self.source_deployment_conversation_name, 'latest_source_deployment_conversation_id': self.latest_source_deployment_conversation_id, 'enable_email_alerts': self.enable_email_alerts, 'email': self.email, 'num_unread_task_instances': self.num_unread_task_instances, 'compute_points_used': self.compute_points_used, 'display_markdown': self.display_markdown, 'requires_new_conversation': self.requires_new_conversation, 'execution_mode': self.execution_mode, 'offer_model_switch': self.offer_model_switch,
                'task_definition': self.task_definition, 'web_app_hostname': self.web_app_hostname, 'trigger_type': self.trigger_type, 'next_run_skipped': self.next_run_skipped, 'skipped_run_time': self.skipped_run_time, 'webhook_url': self.webhook_url, 'push_notifications_enabled': self.push_notifications_enabled, 'skills': self.skills, 'is_rsi': self.is_rsi, 'is_supercomputer': self.is_supercomputer, 'personal_agent_computer_id': self.personal_agent_computer_id, 'current_effort': self.current_effort, 'effort_ladder': self.effort_ladder, 'is_owner': self.is_owner, 'is_org_shared': self.is_org_shared, 'is_directly_shared': self.is_directly_shared, 'is_project_shared': self.is_project_shared, 'shared_permission': self.shared_permission, 'owner_email': self.owner_email, 'resume_request_id': self.resume_request_id, 'hosted_database': self._get_attribute_as_dict(self.hosted_database), 'latest_daemon_task_instance': self._get_attribute_as_dict(self.latest_daemon_task_instance)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
