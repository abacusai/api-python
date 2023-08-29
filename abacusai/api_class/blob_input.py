import dataclasses
import mimetypes

from .abstract import ApiClass


@dataclasses.dataclass
class BlobInput(ApiClass):
    """
    Binary large object input data.

    Args:
        filename (str): The original filename of the blob.
        contents (bytes): The binary contents of the blob.
        mime_type (str): The mime type of the blob.
        size (int): The size of the blob in bytes.
    """
    filename: str
    contents: bytes
    mime_type: str
    size: int

    @classmethod
    def from_local_file(cls, file_path: str) -> 'BlobInput':
        with open(file_path, 'rb') as f:
            contents = f.read()
        return cls.from_contents(contents, filename=file_path)

    @classmethod
    def from_contents(cls, contents: bytes, filename: str = None, mime_type: str = None) -> 'BlobInput':
        if not mime_type and filename:
            mime_type = mimetypes.guess_type(filename)[0]
        return cls(filename=filename, contents=contents, mime_type=mime_type, size=len(contents))
