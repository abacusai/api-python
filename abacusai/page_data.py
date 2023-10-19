from .return_class import AbstractApiClass


class PageData(AbstractApiClass):
    """
        Data extracted from a docstore page.

        Args:
            client (ApiClient): An authenticated API Client instance
            docId (str): Unique Docstore string identifier for the document.
            page (int): The page number. Starts from 0.
            height (int): The height of the page in pixels.
            width (int): The width of the page in pixels.
            pageCount (int): The total number of pages in document.
            pageText (str): The text extracted from the page.
            pageTokenStartOffset (int): The offset of the first token in the page.
            tokenCount (int): The number of tokens in the page.
            tokens (list): The tokens in the page.
    """

    def __init__(self, client, docId=None, page=None, height=None, width=None, pageCount=None, pageText=None, pageTokenStartOffset=None, tokenCount=None, tokens=None):
        super().__init__(client, None)
        self.doc_id = docId
        self.page = page
        self.height = height
        self.width = width
        self.page_count = pageCount
        self.page_text = pageText
        self.page_token_start_offset = pageTokenStartOffset
        self.token_count = tokenCount
        self.tokens = tokens

    def __repr__(self):
        repr_dict = {f'doc_id': repr(self.doc_id), f'page': repr(self.page), f'height': repr(self.height), f'width': repr(self.width), f'page_count': repr(self.page_count), f'page_text': repr(
            self.page_text), f'page_token_start_offset': repr(self.page_token_start_offset), f'token_count': repr(self.token_count), f'tokens': repr(self.tokens)}
        class_name = "PageData"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'doc_id': self.doc_id, 'page': self.page, 'height': self.height, 'width': self.width, 'page_count': self.page_count,
                'page_text': self.page_text, 'page_token_start_offset': self.page_token_start_offset, 'token_count': self.token_count, 'tokens': self.tokens}
        return {key: value for key, value in resp.items() if value is not None}
