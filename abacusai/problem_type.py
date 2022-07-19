from .return_class import AbstractApiClass


class ProblemType(AbstractApiClass):
    """
        Description of a problem type which is the common underlying problem for different use cases.

        Args:
            client (ApiClient): An authenticated API Client instance
            problemType (str): 
            featureGroupTypes (list of string): The feature group types can be trained on for this problem type
    """

    def __init__(self, client, problemType=None, featureGroupTypes=None):
        super().__init__(client, None)
        self.problem_type = problemType
        self.feature_group_types = featureGroupTypes

    def __repr__(self):
        return f"ProblemType(problem_type={repr(self.problem_type)},\n  feature_group_types={repr(self.feature_group_types)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'problem_type': self.problem_type, 'feature_group_types': self.feature_group_types}
