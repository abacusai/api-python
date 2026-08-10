from .return_class import AbstractApiClass


class MediaProjectRenderState(AbstractApiClass):
    """
        Media Project Render State — a project's export markers, without the project itself.

        Args:
            client (ApiClient): An authenticated API Client instance
            mediaProjectId (id): The ID of the media project
            activeRender (dict): The render in flight (stage, request id, start times), absent when idle
            lastRenderStats (dict): The previous successful render's actuals, which seed the time estimate
    """

    def __init__(self, client, mediaProjectId=None, activeRender=None, lastRenderStats=None):
        super().__init__(client, None)
        self.media_project_id = mediaProjectId
        self.active_render = activeRender
        self.last_render_stats = lastRenderStats
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'media_project_id': repr(self.media_project_id), f'active_render': repr(
            self.active_render), f'last_render_stats': repr(self.last_render_stats)}
        class_name = "MediaProjectRenderState"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'media_project_id': self.media_project_id,
                'active_render': self.active_render, 'last_render_stats': self.last_render_stats}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
