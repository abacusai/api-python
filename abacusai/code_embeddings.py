from .return_class import AbstractApiClass


class CodeEmbeddings(AbstractApiClass):
    """
        Code embeddings

        Args:
            client (ApiClient): An authenticated API Client instance
            embeddings (dict): A dictionary mapping the file name to its embeddings.
    """

    def __init__(self, client, embeddings=None):
        super().__init__(client, None)
        self.embeddings = embeddings
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'embeddings': repr(self.embeddings)}
        class_name = "CodeEmbeddings"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'embeddings': self.embeddings}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
