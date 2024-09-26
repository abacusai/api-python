import dataclasses
import mimetypes

from .abstract import ApiClass


@dataclasses.dataclass
class Blob(ApiClass):
    """
    An object for storing and passing file data.
    In AI Agents, if a function accepts file upload as an argument, the uploaded file is passed as a Blob object. If a function returns a Blob object, it will be rendered as a file download.

    Args:
        contents (bytes): The binary contents of the blob.
        mime_type (str): The mime type of the blob.
        filename (str): The original filename of the blob.
        size (int): The size of the blob in bytes.
    """
    filename: str
    contents: bytes
    mime_type: str
    size: int

    def __init__(self, contents: bytes, mime_type: str = None, filename: str = None, size: int = None):
        if contents is None or not isinstance(contents, bytes):
            raise ValueError('contents must be a valid bytes object')
        if mime_type is None:
            try:
                if filename:
                    mime_type = mimetypes.guess_type(filename)[0]
                else:
                    import magic
                    mime_type = magic.Magic(mime=True).from_buffer(contents)
            except Exception:
                pass
        else:
            if not isinstance(mime_type, str):
                raise ValueError('mime_type must be a valid string')

        self.filename = filename
        self.contents = contents
        self.mime_type = mime_type
        self.size = size or len(contents)

    @classmethod
    def from_local_file(cls, file_path: str) -> 'Blob':
        with open(file_path, 'rb') as f:
            contents = f.read()
        return cls.from_contents(contents, filename=file_path)

    @classmethod
    def from_contents(cls, contents: bytes, filename: str = None, mime_type: str = None) -> 'Blob':
        return cls(filename=filename, contents=contents, mime_type=mime_type, size=len(contents))


@dataclasses.dataclass
class BlobInput(Blob):
    """
    An object for storing and passing file data.
    In AI Agents, if a function accepts file upload as an argument, the uploaded file is passed as a BlobInput object.

    Args:
        filename (str): The original filename of the blob.
        contents (bytes): The binary contents of the blob.
        mime_type (str): The mime type of the blob.
        size (int): The size of the blob in bytes.
    """

    def __init__(self, filename: str = None, contents: bytes = None, mime_type: str = None, size: int = None):
        super().__init__(contents, mime_type, filename, size)
