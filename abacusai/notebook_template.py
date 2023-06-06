from .return_class import AbstractApiClass


class NotebookTemplate(AbstractApiClass):
    """
        A template for notebooks.

        Args:
            client (ApiClient): An authenticated API Client instance
            notebookTemplateId (str): The ID of the notebook template
            name (str): The name of the notebook template
            description (str): The description of the notebook template
            createdAt (str): The date and time which the notebook template was created.
            updatedAt (str): The date and time which the notebook template was updated.
            templateType (str): The type of the notebook template
            filename (str): The file name of the notebook template
    """

    def __init__(self, client, notebookTemplateId=None, name=None, description=None, createdAt=None, updatedAt=None, templateType=None, filename=None):
        super().__init__(client, notebookTemplateId)
        self.notebook_template_id = notebookTemplateId
        self.name = name
        self.description = description
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.template_type = templateType
        self.filename = filename

    def __repr__(self):
        return f"NotebookTemplate(notebook_template_id={repr(self.notebook_template_id)},\n  name={repr(self.name)},\n  description={repr(self.description)},\n  created_at={repr(self.created_at)},\n  updated_at={repr(self.updated_at)},\n  template_type={repr(self.template_type)},\n  filename={repr(self.filename)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'notebook_template_id': self.notebook_template_id, 'name': self.name, 'description': self.description, 'created_at': self.created_at, 'updated_at': self.updated_at, 'template_type': self.template_type, 'filename': self.filename}
