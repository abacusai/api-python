from .return_class import AbstractApiClass


class PointInTimeGroupFeature(AbstractApiClass):
    """
        A point in time group feature

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name of the feature
            expression (str): SQL Aggregate expression which can convert a sequence of rows into a scalar value.
    """

    def __init__(self, client, name=None, expression=None):
        super().__init__(client, None)
        self.name = name
        self.expression = expression

    def __repr__(self):
        return f"PointInTimeGroupFeature(name={repr(self.name)},\n  expression={repr(self.expression)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'expression': self.expression}
