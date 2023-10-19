from .return_class import AbstractApiClass


class CodeSource(AbstractApiClass):
    """
        Code source for python-based custom feature groups and models

        Args:
            client (ApiClient): An authenticated API Client instance
            sourceType (str): The type of the source, one of TEXT, PYTHON, FILE_UPLOAD, or APPLICATION_CONNECTOR
            sourceCode (str): If the type of the source is TEXT, the raw text of the function
            applicationConnectorId (str): The Application Connector to fetch the code from
            applicationConnectorInfo (str): Args passed to the application connector to fetch the code
            packageRequirements (list): The pip package dependencies required to run the code
            status (str): The status of the code and validations
            error (str): If the status is failed, an error message describing what went wrong
            publishingMsg (dict): Warnings in the source code
            moduleDependencies (list): The list of internal modules dependencies required to run the code
    """

    def __init__(self, client, sourceType=None, sourceCode=None, applicationConnectorId=None, applicationConnectorInfo=None, packageRequirements=None, status=None, error=None, publishingMsg=None, moduleDependencies=None):
        super().__init__(client, None)
        self.source_type = sourceType
        self.source_code = sourceCode
        self.application_connector_id = applicationConnectorId
        self.application_connector_info = applicationConnectorInfo
        self.package_requirements = packageRequirements
        self.status = status
        self.error = error
        self.publishing_msg = publishingMsg
        self.module_dependencies = moduleDependencies

    def __repr__(self):
        repr_dict = {f'source_type': repr(self.source_type), f'source_code': repr(self.source_code), f'application_connector_id': repr(self.application_connector_id), f'application_connector_info': repr(
            self.application_connector_info), f'package_requirements': repr(self.package_requirements), f'status': repr(self.status), f'error': repr(self.error), f'publishing_msg': repr(self.publishing_msg), f'module_dependencies': repr(self.module_dependencies)}
        class_name = "CodeSource"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'source_type': self.source_type, 'source_code': self.source_code, 'application_connector_id': self.application_connector_id, 'application_connector_info': self.application_connector_info,
                'package_requirements': self.package_requirements, 'status': self.status, 'error': self.error, 'publishing_msg': self.publishing_msg, 'module_dependencies': self.module_dependencies}
        return {key: value for key, value in resp.items() if value is not None}

    def import_as_cell(self):
        """
        Adds the source code as an unexecuted cell in the notebook.
        """
        if not self.source_code:
            print('No source code to import.')
        from IPython.core.getipython import get_ipython
        try:
            shell = get_ipython()
            shell.set_next_input(self.source_code)
        except Exception:
            print('Unable to import source code as cells.')
