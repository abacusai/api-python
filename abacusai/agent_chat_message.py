from .return_class import AbstractApiClass


class AgentChatMessage(AbstractApiClass):
    """
        A single chat message with Agent Chat.

        Args:
            client (ApiClient): An authenticated API Client instance
            role (str): The role of the message sender
            text (list[dict]): A list of text segments for the message
            docIds (list[str]): A list of IDs of the uploaded document if the message has
            keywordArguments (dict): User message only. A dictionary of keyword arguments used to generate response.
            segments (list[dict]): A list of segments for the message
            streamedData (str): The streamed data for the message
            streamedSectionData (list): A list of streamed section data for the message
            agentWorkflowNodeId (str): The workflow node name associated with the agent response.
    """

    def __init__(self, client, role=None, text=None, docIds=None, keywordArguments=None, segments=None, streamedData=None, streamedSectionData=None, agentWorkflowNodeId=None):
        super().__init__(client, None)
        self.role = role
        self.text = text
        self.doc_ids = docIds
        self.keyword_arguments = keywordArguments
        self.segments = segments
        self.streamed_data = streamedData
        self.streamed_section_data = streamedSectionData
        self.agent_workflow_node_id = agentWorkflowNodeId
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'role': repr(self.role), f'text': repr(self.text), f'doc_ids': repr(self.doc_ids), f'keyword_arguments': repr(self.keyword_arguments), f'segments': repr(
            self.segments), f'streamed_data': repr(self.streamed_data), f'streamed_section_data': repr(self.streamed_section_data), f'agent_workflow_node_id': repr(self.agent_workflow_node_id)}
        class_name = "AgentChatMessage"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'role': self.role, 'text': self.text, 'doc_ids': self.doc_ids, 'keyword_arguments': self.keyword_arguments, 'segments': self.segments,
                'streamed_data': self.streamed_data, 'streamed_section_data': self.streamed_section_data, 'agent_workflow_node_id': self.agent_workflow_node_id}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
