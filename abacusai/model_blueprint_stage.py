from .return_class import AbstractApiClass


class ModelBlueprintStage(AbstractApiClass):
    """
        A stage in the model blueprint export process.

        Args:
            client (ApiClient): An authenticated API Client instance
            stageName (str): The name of the stage.
            displayName (str): The display name of the stage.
            description (str): The description of the stage.
            params (dict): The parameters for the stage.
            predecessors (list): A list of stages that occur directly before this stage.
    """

    def __init__(self, client, stageName=None, displayName=None, description=None, params=None, predecessors=None):
        super().__init__(client, None)
        self.stage_name = stageName
        self.display_name = displayName
        self.description = description
        self.params = params
        self.predecessors = predecessors

    def __repr__(self):
        return f"ModelBlueprintStage(stage_name={repr(self.stage_name)},\n  display_name={repr(self.display_name)},\n  description={repr(self.description)},\n  params={repr(self.params)},\n  predecessors={repr(self.predecessors)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'stage_name': self.stage_name, 'display_name': self.display_name, 'description': self.description, 'params': self.params, 'predecessors': self.predecessors}
