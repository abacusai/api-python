from .model_blueprint_stage import ModelBlueprintStage
from .return_class import AbstractApiClass


class ModelBlueprintExport(AbstractApiClass):
    """
        Model Blueprint

        Args:
            client (ApiClient): An authenticated API Client instance
            modelVersion (str): Version of the model that the blueprint is for.
            currentTrainingConfig (dict): The current training configuration for the model. It can be used to get training configs and train a new model
            modelBlueprintStages (ModelBlueprintStage): The stages of the model blueprint. Each one includes the stage name, display name, description, parameters, and predecessors.
    """

    def __init__(self, client, modelVersion=None, currentTrainingConfig=None, modelBlueprintStages={}):
        super().__init__(client, None)
        self.model_version = modelVersion
        self.current_training_config = currentTrainingConfig
        self.model_blueprint_stages = client._build_class(
            ModelBlueprintStage, modelBlueprintStages)

    def __repr__(self):
        return f"ModelBlueprintExport(model_version={repr(self.model_version)},\n  current_training_config={repr(self.current_training_config)},\n  model_blueprint_stages={repr(self.model_blueprint_stages)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'model_version': self.model_version, 'current_training_config': self.current_training_config, 'model_blueprint_stages': self._get_attribute_as_dict(self.model_blueprint_stages)}
