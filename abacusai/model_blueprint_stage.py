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
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'stage_name': repr(self.stage_name), f'display_name': repr(self.display_name), f'description': repr(
            self.description), f'params': repr(self.params), f'predecessors': repr(self.predecessors)}
        class_name = "ModelBlueprintStage"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'stage_name': self.stage_name, 'display_name': self.display_name,
                'description': self.description, 'params': self.params, 'predecessors': self.predecessors}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
