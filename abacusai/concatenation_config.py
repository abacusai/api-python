from .return_class import AbstractApiClass


class ConcatenationConfig(AbstractApiClass):
    """
        Feature Group Concatenation Config

        Args:
            client (ApiClient): An authenticated API Client instance
            concatenatedTable (str): The feature group to concatenate with the destination feature group.
            mergeType (str): The type of merge to perform, either `UNION` or `INTERSECTION`.
            replaceUntilTimestamp (int): The Unix timestamp to specify the point up to which data from the source feature group will be replaced.
            skipMaterialize (bool): If `True`, the concatenated feature group will not be materialized.
    """

    def __init__(self, client, concatenatedTable=None, mergeType=None, replaceUntilTimestamp=None, skipMaterialize=None):
        super().__init__(client, None)
        self.concatenated_table = concatenatedTable
        self.merge_type = mergeType
        self.replace_until_timestamp = replaceUntilTimestamp
        self.skip_materialize = skipMaterialize

    def __repr__(self):
        repr_dict = {f'concatenated_table': repr(self.concatenated_table), f'merge_type': repr(
            self.merge_type), f'replace_until_timestamp': repr(self.replace_until_timestamp), f'skip_materialize': repr(self.skip_materialize)}
        class_name = "ConcatenationConfig"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'concatenated_table': self.concatenated_table, 'merge_type': self.merge_type,
                'replace_until_timestamp': self.replace_until_timestamp, 'skip_materialize': self.skip_materialize}
        return {key: value for key, value in resp.items() if value is not None}
