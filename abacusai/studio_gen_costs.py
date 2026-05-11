from .return_class import AbstractApiClass


class StudioGenCosts(AbstractApiClass):
    """
        Credit cost estimation data for all Studio models.

        Args:
            client (ApiClient): An authenticated API Client instance
            videoCosts (dict): Cost lookup tables for video generation models
            imageCosts (dict): Cost lookup tables for image generation models
            lipSyncCosts (dict): Cost lookup tables for lip sync models
            overhead (int): Fixed credit overhead per generation (agentic loop cost)
    """

    def __init__(self, client, videoCosts=None, imageCosts=None, lipSyncCosts=None, overhead=None):
        super().__init__(client, None)
        self.video_costs = videoCosts
        self.image_costs = imageCosts
        self.lip_sync_costs = lipSyncCosts
        self.overhead = overhead
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'video_costs': repr(self.video_costs), f'image_costs': repr(
            self.image_costs), f'lip_sync_costs': repr(self.lip_sync_costs), f'overhead': repr(self.overhead)}
        class_name = "StudioGenCosts"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'video_costs': self.video_costs, 'image_costs': self.image_costs,
                'lip_sync_costs': self.lip_sync_costs, 'overhead': self.overhead}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
