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

    def __repr__(self):
        return f"DataConsistencyDuplication(total_count={repr(self.total_count)},\n  num_duplicates={repr(self.num_duplicates)},\n  sample={repr(self.sample)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'total_count': self.total_count, 'num_duplicates': self.num_duplicates, 'sample': self._get_attribute_as_dict(self.sample)}
