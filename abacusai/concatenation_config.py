from .return_class import AbstractApiClass


class ConcatenationConfig(AbstractApiClass):
    """
        Feature Group Concatenation Config

        Args:
            client (ApiClient): An authenticated API Client instance
            concatenatedTable (str): The feature group to concatenate with the destination feature group.
            mergeType (str): UNION or INTERSECTION
            replaceUntilTimestamp (int): The unix timestamp to specify the point till which we will replace data from the source feature group.
            skipMaterialize (bool): If true, will not materialize the concatenated feature group
    """

    def __init__(self, client, concatenatedTable=None, mergeType=None, replaceUntilTimestamp=None, skipMaterialize=None):
        super().__init__(client, None)
        self.concatenated_table = concatenatedTable
        self.merge_type = mergeType
        self.replace_until_timestamp = replaceUntilTimestamp
        self.skip_materialize = skipMaterialize

    def __repr__(self):
        return f"ConcatenationConfig(concatenated_table={repr(self.concatenated_table)},\n  merge_type={repr(self.merge_type)},\n  replace_until_timestamp={repr(self.replace_until_timestamp)},\n  skip_materialize={repr(self.skip_materialize)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'concatenated_table': self.concatenated_table, 'merge_type': self.merge_type, 'replace_until_timestamp': self.replace_until_timestamp, 'skip_materialize': self.skip_materialize}