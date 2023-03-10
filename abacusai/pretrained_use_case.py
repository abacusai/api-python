from .return_class import AbstractApiClass


class PretrainedUseCase(AbstractApiClass):
    """
        Pretrained use case. Used by Abacus.AI internally.

        Args:
            client (ApiClient): An authenticated API Client instance
            useCase (str): Name of the use case, which is the key for the use case in the system.
            displayName (str): The name to show to external users.
            description (str): A detailed description of the use case.
            predictionApi (str): Which prediction api should be used for this use case.
            predictionUiDisplayType (str): The component type to show this use case's prediction dashboard.
            imgSrc (str): The source of the image for this use case.
    """

    def __init__(self, client, useCase=None, displayName=None, description=None, predictionApi=None, predictionUiDisplayType=None, imgSrc=None):
        super().__init__(client, None)
        self.use_case = useCase
        self.display_name = displayName
        self.description = description
        self.prediction_api = predictionApi
        self.prediction_ui_display_type = predictionUiDisplayType
        self.img_src = imgSrc

    def __repr__(self):
        return f"PretrainedUseCase(use_case={repr(self.use_case)},\n  display_name={repr(self.display_name)},\n  description={repr(self.description)},\n  prediction_api={repr(self.prediction_api)},\n  prediction_ui_display_type={repr(self.prediction_ui_display_type)},\n  img_src={repr(self.img_src)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'use_case': self.use_case, 'display_name': self.display_name, 'description': self.description, 'prediction_api': self.prediction_api, 'prediction_ui_display_type': self.prediction_ui_display_type, 'img_src': self.img_src}
