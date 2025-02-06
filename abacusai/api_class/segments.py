import dataclasses
import uuid
from typing import Any, List

from . import enums
from .abstract import ApiClass


@dataclasses.dataclass
class ResponseSection(ApiClass):
    """
    A response section that an agent can return to render specific UI elements.

    Args:
        type (ResponseSectionType): The type of the response.
        id (str): The section key of the segment.
    """

    type: enums.ResponseSectionType
    id: str

    def __post_init__(self):
        self.message_id = str(uuid.uuid4())

    def to_dict(self):
        return {**{k: v.value if isinstance(v, enums.Enum) else v for k, v in dataclasses.asdict(self).items() if not k.startswith('_')}, 'message_id': self.message_id}


Segment = ResponseSection


@dataclasses.dataclass
class AgentFlowButtonResponseSection(ResponseSection):
    """
    A response section that an AI Agent can return to render a button.

    Args:
        label (str): The label of the button.
        agent_workflow_node_name (str): The workflow start node to be executed when the button is clicked.
    """

    label: str
    agent_workflow_node_name: str

    def __init__(self, label: str, agent_workflow_node_name: str, section_key: str = None):
        super().__init__(type=enums.ResponseSectionType.AGENT_FLOW_BUTTON, id=section_key)
        self.label = label
        self.agent_workflow_node_name = agent_workflow_node_name


@dataclasses.dataclass
class ImageUrlResponseSection(ResponseSection):
    """
    A response section that an agent can return to render an image.

    Args:
        url (str): The url of the image to be displayed.
        height (int): The height of the image.
        width (int): The width of the image.
    """

    url: str
    height: int
    width: int

    def __init__(self, url: str, height: int, width: int, section_key: str = None):
        super().__init__(type=enums.ResponseSectionType.IMAGE_URL, id=section_key)
        self.url = url
        self.height = height
        self.width = width


@dataclasses.dataclass
class TextResponseSection(ResponseSection):
    """
    A response section that an agent can return to render text.

    Args:
        segment (str): The text to be displayed.
    """

    segment: str

    def __init__(self, text: str, section_key: str = None):
        super().__init__(type=enums.ResponseSectionType.TEXT, id=section_key)
        self.segment = text


@dataclasses.dataclass
class RuntimeSchemaResponseSection(ResponseSection):
    """
    A segment that an agent can return to render json and ui schema in react-jsonschema-form format for workflow nodes.
    This is primarily used to generate dynamic forms at runtime. If a node returns a runtime schema variable, the UI will render the form upon node execution.

    Args:
        json_schema (dict): json schema in RJSF format.
        ui_schema (dict): ui schema in RJSF format.
    """

    json_schema: dict
    ui_schema: dict

    def __init__(self, json_schema: dict, ui_schema: dict = None, schema_prop: str = None):
        super().__init__(type=enums.ResponseSectionType.RUNTIME_SCHEMA, id=schema_prop)
        self.json_schema = json_schema
        self.ui_schema = ui_schema or {}


@dataclasses.dataclass
class CodeResponseSection(ResponseSection):
    """
    A response section that an agent can return to render code.

    Args:
        code (str): The code to be displayed.
        language (CodeLanguage): The language of the code.
    """

    code: str
    language: enums.CodeLanguage

    def __init__(self, code: str, language: enums.CodeLanguage, section_key: str = None):
        super().__init__(enums.ResponseSectionType.CODE, id=section_key)
        self.code = code
        self.language = language


@dataclasses.dataclass
class Base64ImageResponseSection(ResponseSection):
    """
    A response section that an agent can return to render a base64 image.

    Args:
        b64_image (str): The base64 image to be displayed.
    """

    b64_image: str

    def __init__(self, b64_image: str, section_key: str = None):
        super().__init__(enums.ResponseSectionType.BASE64_IMAGE, id=section_key)
        self.b64_image = b64_image


@dataclasses.dataclass
class CollapseResponseSection(ResponseSection):
    """
    A response section that an agent can return to render a collapsible component.

    Args:
        title (str): The title of the collapsible component.
        content (ResponseSection): The response section constituting the content of collapsible component
    """

    title: str
    content: ResponseSection

    def __init__(self, title: str, content: ResponseSection, section_key: str = None):
        super().__init__(enums.ResponseSectionType.COLLAPSIBLE_COMPONENT, id=section_key)
        self.title = title
        self.content = content

    def to_dict(self):
        return {
            'title': self.title,
            'content': self.content.to_dict(),
            'type': self.type.value,
            'id': self.id
        }


@dataclasses.dataclass
class ListResponseSection(ResponseSection):
    """
    A response section that an agent can return to render a list.

    Args:
        items (List[str]): The list items to be displayed.
    """

    items: List[str]

    def __init__(self, items: List[str], section_key: str = None):
        super().__init__(enums.ResponseSectionType.LIST, id=section_key)
        self.items = items


@dataclasses.dataclass
class ChartResponseSection(ResponseSection):
    """
    A response section that an agent can return to render a chart.

    Args:
        chart (dict): The chart to be displayed.
    """

    chart: dict

    def __init__(self, chart: dict, section_key: str = None):
        super().__init__(enums.ResponseSectionType.CHART, id=section_key)
        self.chart = chart


@dataclasses.dataclass
class DataframeResponseSection(ResponseSection):
    """
    A response section that an agent can return to render a pandas dataframe.
    Args:
        df (pandas.DataFrame): The dataframe to be displayed.
        header (str): Heading of the table to be displayed.
    """

    df: Any
    header: str

    def __init__(self, df: Any, header: str = None, section_key: str = None):
        if type(df).__name__ != 'DataFrame':
            raise ValueError('Invalid dataframe instance. Argument "df" must be a pandas DataFrame.')
        super().__init__(enums.ResponseSectionType.TABLE, id=section_key)
        if not header:
            header = 'Data' if df.shape[0] <= 50 else 'Truncated Data Preview'
        self.df = df
        self.header = header
