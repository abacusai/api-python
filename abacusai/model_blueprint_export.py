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
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'model_version': repr(self.model_version), f'current_training_config': repr(
            self.current_training_config), f'model_blueprint_stages': repr(self.model_blueprint_stages)}
        class_name = "ModelBlueprintExport"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'model_version': self.model_version, 'current_training_config': self.current_training_config,
                'model_blueprint_stages': self._get_attribute_as_dict(self.model_blueprint_stages)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
