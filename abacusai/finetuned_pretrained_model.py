from .return_class import AbstractApiClass


class FinetunedPretrainedModel(AbstractApiClass):
    """
        A finetuned pretrained model

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The user-friendly name for the model.
            finetunedPretrainedModelId (str): The unique identifier of the model.
            finetunedPretrainedModelVersion (str): The unique identifier of the model version.
            createdAt (str): When the finetuned pretrained model was created.
            updatedAt (str): When the finetuned pretrained model was last updated.
            config (dict): The finetuned pretrained model configuration
            baseModel (str): The pretrained base model for fine tuning
            finetuningDatasetVersion (str): The finetuned dataset instance id of the model.
            status (str): The current status of the finetuned pretrained model.
            error (str): Relevant error if the status is FAILED.
    """

    def __init__(self, client, name=None, finetunedPretrainedModelId=None, finetunedPretrainedModelVersion=None, createdAt=None, updatedAt=None, config=None, baseModel=None, finetuningDatasetVersion=None, status=None, error=None):
        super().__init__(client, finetunedPretrainedModelId)
        self.name = name
        self.finetuned_pretrained_model_id = finetunedPretrainedModelId
        self.finetuned_pretrained_model_version = finetunedPretrainedModelVersion
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.config = config
        self.base_model = baseModel
        self.finetuning_dataset_version = finetuningDatasetVersion
        self.status = status
        self.error = error
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'finetuned_pretrained_model_id': repr(self.finetuned_pretrained_model_id), f'finetuned_pretrained_model_version': repr(self.finetuned_pretrained_model_version), f'created_at': repr(
            self.created_at), f'updated_at': repr(self.updated_at), f'config': repr(self.config), f'base_model': repr(self.base_model), f'finetuning_dataset_version': repr(self.finetuning_dataset_version), f'status': repr(self.status), f'error': repr(self.error)}
        class_name = "FinetunedPretrainedModel"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'finetuned_pretrained_model_id': self.finetuned_pretrained_model_id, 'finetuned_pretrained_model_version': self.finetuned_pretrained_model_version, 'created_at': self.created_at,
                'updated_at': self.updated_at, 'config': self.config, 'base_model': self.base_model, 'finetuning_dataset_version': self.finetuning_dataset_version, 'status': self.status, 'error': self.error}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
