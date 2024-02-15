from .return_class import AbstractApiClass


class DataConsistencyDuplication(AbstractApiClass):
    """
        Data Consistency for duplication within data

        Args:
            client (ApiClient): An authenticated API Client instance
            totalCount (int): Total count of rows in data.
            numDuplicates (int): Number of Duplicates based on primary keys in data.
            sample (FeatureRecord): A list of dicts enumerating rows the rows that contained duplications in primary keys.
    """

    def __init__(self, client, totalCount=None, numDuplicates=None, sample={}):
        super().__init__(client, None)
        self.total_count = totalCount
        self.num_duplicates = numDuplicates
        self.sample = client._build_class(FeatureRecord, sample)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'total_count': repr(self.total_count), f'num_duplicates': repr(
            self.num_duplicates), f'sample': repr(self.sample)}
        class_name = "DataConsistencyDuplication"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'total_count': self.total_count, 'num_duplicates': self.num_duplicates,
                'sample': self._get_attribute_as_dict(self.sample)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
