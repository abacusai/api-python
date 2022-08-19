from .return_class import AbstractApiClass


class ProblemType(AbstractApiClass):
    """
        Description of a problem type which is the common underlying problem for different use cases.

        Args:
            client (ApiClient): An authenticated API Client instance
            problemType (str): 
            requiredFeatureGroupType (str): The required feature group types to train for this problem type
            optionalFeatureGroupTypes (list of string): The optional feature group types can be used to train for this problem type
            useCasesSupportCustomAlgorithm (list): A list of use cases that support custom algorithms
    """

    def __init__(self, client, problemType=None, requiredFeatureGroupType=None, optionalFeatureGroupTypes=None, useCasesSupportCustomAlgorithm=None):
        super().__init__(client, None)
        self.problem_type = problemType
        self.required_feature_group_type = requiredFeatureGroupType
        self.optional_feature_group_types = optionalFeatureGroupTypes
        self.use_cases_support_custom_algorithm = useCasesSupportCustomAlgorithm

    def __repr__(self):
        return f"ProblemType(problem_type={repr(self.problem_type)},\n  required_feature_group_type={repr(self.required_feature_group_type)},\n  optional_feature_group_types={repr(self.optional_feature_group_types)},\n  use_cases_support_custom_algorithm={repr(self.use_cases_support_custom_algorithm)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'problem_type': self.problem_type, 'required_feature_group_type': self.required_feature_group_type, 'optional_feature_group_types': self.optional_feature_group_types, 'use_cases_support_custom_algorithm': self.use_cases_support_custom_algorithm}
