from .return_class import AbstractApiClass


class PointInTimeFeatureInfo(AbstractApiClass):
    """
        A point-in-time infos for a feature

        Args:
            client (ApiClient): An authenticated API Client instance
            expression (str): SQL aggregate expression which can convert a sequence of rows into a scalar value.
            groupName (str): The group name this point-in-time feature belongs to.
    """

    def __init__(self, client, expression=None, groupName=None):
        super().__init__(client, None)
        self.expression = expression
        self.group_name = groupName
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'expression': repr(
            self.expression), f'group_name': repr(self.group_name)}
        class_name = "PointInTimeFeatureInfo"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'expression': self.expression, 'group_name': self.group_name}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
