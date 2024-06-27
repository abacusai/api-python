import dataclasses
from typing import List

from . import enums
from .abstract import ApiClass


@dataclasses.dataclass
class Attachment(ApiClass):
    """
    An attachment that an agent can return to render attachments.

    Args:
        filename (str): The name of the file.
        mime_type (str): The MIME type of the file.
        attachment_id (str): The ID of the attachment.
    """

    filename: str
    mime_type: str
    attachment_id: str

    def to_dict(self):
        return {
            'type': 'attachment',
            'filename': self.filename,
            'mime_type': self.mime_type,
            'attachment_id': self.attachment_id
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(filename=data['filename'], mime_type=data['mime_type'], attachment_id=data['attachment_id'])


@dataclasses.dataclass
class Segment(ApiClass):
    """
    A segment that an agent can return to render specific UI elements.

    Args:
        type (SegmentType): The type of the segment.
        id (str): The section key of the segment.
    """

    type: enums.SegmentType
    id: str

    def to_dict(self):
        return {k: v.value if isinstance(v, enums.Enum) else v for k, v in dataclasses.asdict(self).items()}


@dataclasses.dataclass
class AttachmentsSegment(Segment):
    """
    A segment that an agent can return to render attachments.

    Args:
        attachments (List[Attachment]): The list of attachments to be displayed.
    """

    attachments: List[Attachment]

    def __init__(self, attachments: List[Attachment], section_key: str = None):
        super().__init__(type=enums.SegmentType.ATTACHMENTS, id=section_key)
        self.attachments = attachments

    def to_dict(self):
        return {
            'type': self.type.value,
            'id': self.id,
            'attachments': [attachment.to_dict() for attachment in self.attachments]
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(attachments=[Attachment.from_dict(attachment) for attachment in data['attachments']])


@dataclasses.dataclass
class AgentFlowButtonSegment(Segment):
    """
    A segment that an AI Agent can return to render a button.

    Args:
        label (str): The label of the button.
        agent_workflow_node_name (str): The workflow start node to be executed when the button is clicked.
    """

    label: str
    agent_workflow_node_name: str

    def __init__(self, label: str, agent_workflow_node_name: str, section_key: str = None):
        super().__init__(type=enums.SegmentType.AGENT_FLOW_BUTTON, id=section_key)
        self.label = label
        self.agent_workflow_node_name = agent_workflow_node_name


@dataclasses.dataclass
class ImageUrlSegment(Segment):
    """
    A segment that an agent can return to render an image.

    Args:
        segment (str): The url of the image to be displayed.
    """

    segment: str

    def __init__(self, url: str, section_key: str = None):
        super().__init__(type=enums.SegmentType.IMAGE_URL, id=section_key)
        self.segment = url


@dataclasses.dataclass
class TextSegment(Segment):
    """
    A segment that an agent can return to render text.

    Args:
        segment (str): The text to be displayed.
    """

    segment: str

    def __init__(self, text: str, section_key: str = None):
        super().__init__(type=enums.SegmentType.TEXT, id=section_key)
        self.segment = text
