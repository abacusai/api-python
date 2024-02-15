from .cpu_gpu_memory_specs import CpuGpuMemorySpecs
from .return_class import AbstractApiClass


class MemoryOptions(AbstractApiClass):
    """
        The overall memory options for executing a job

        Args:
            client (ApiClient): An authenticated API Client instance
            cpu (CpuGpuMemorySpecs): Contains information about the default CPU and list of CPU memory & size options
            gpu (CpuGpuMemorySpecs): Contains information about the default GPU and list of GPU memory & size options
    """

    def __init__(self, client, cpu={}, gpu={}):
        super().__init__(client, None)
        self.cpu = client._build_class(CpuGpuMemorySpecs, cpu)
        self.gpu = client._build_class(CpuGpuMemorySpecs, gpu)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'cpu': repr(self.cpu), f'gpu': repr(self.gpu)}
        class_name = "MemoryOptions"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'cpu': self._get_attribute_as_dict(
            self.cpu), 'gpu': self._get_attribute_as_dict(self.gpu)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
