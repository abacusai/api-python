from .return_class import AbstractApiClass


class CpuGpuMemorySpecs(AbstractApiClass):
    """
        Includes the memory specs of the CPU/GPU

        Args:
            client (ApiClient): An authenticated API Client instance
            default (int): the default memory size for the processing unit
            data (list): the list of memory sizes for the processing unit
    """

    def __init__(self, client, default=None, data=None):
        super().__init__(client, None)
        self.default = default
        self.data = data
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'default': repr(self.default), f'data': repr(self.data)}
        class_name = "CpuGpuMemorySpecs"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'default': self.default, 'data': self.data}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
