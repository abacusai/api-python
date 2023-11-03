from .return_class import AbstractApiClass


class FinetunedPretrainedModel(AbstractApiClass):
    """
        A finetuned pretrained model

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The user-friendly name for the model.
            modelId (str): The unique identifier of the model.
            createdAt (str): When the finetuned pretrained model was created.
            updatedAt (str): When the finetuned pretrained model was last updated.
            config (dict): The finetuned pretrained model configuration
            baseModel (str): The pretrained base model for fine tuning
            finetuningDatasetId (str): The finetuned dataset ids of the model.
    """

    def __init__(self, client, name=None, modelId=None, createdAt=None, updatedAt=None, config=None, baseModel=None, finetuningDatasetId=None):
        super().__init__(client, None)
        self.name = name
        self.model_id = modelId
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.config = config
        self.base_model = baseModel
        self.finetuning_dataset_id = finetuningDatasetId

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'model_id': repr(self.model_id), f'created_at': repr(self.created_at), f'updated_at': repr(
            self.updated_at), f'config': repr(self.config), f'base_model': repr(self.base_model), f'finetuning_dataset_id': repr(self.finetuning_dataset_id)}
        class_name = "FinetunedPretrainedModel"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'model_id': self.model_id, 'created_at': self.created_at, 'updated_at': self.updated_at,
                'config': self.config, 'base_model': self.base_model, 'finetuning_dataset_id': self.finetuning_dataset_id}
        return {key: value for key, value in resp.items() if value is not None}
