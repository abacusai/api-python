from .return_class import AbstractApiClass


class ProjectMemory(AbstractApiClass):
    """
        The shared memory files of an AGENT_COORDINATOR project (the project's Context card)

        Args:
            client (ApiClient): An authenticated API Client instance
            progress (str): PROGRESS.md, what was done run by run.
            learnings (str): LEARNINGS.md, what turned out to be true about the codebase, data or domain.
            decisions (str): DECISIONS.md, choices made and why.
    """

    def __init__(self, client, progress=None, learnings=None, decisions=None):
        super().__init__(client, None)
        self.progress = progress
        self.learnings = learnings
        self.decisions = decisions
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'progress': repr(self.progress), f'learnings': repr(
            self.learnings), f'decisions': repr(self.decisions)}
        class_name = "ProjectMemory"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'progress': self.progress,
                'learnings': self.learnings, 'decisions': self.decisions}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
