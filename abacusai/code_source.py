from .return_class import AbstractApiClass


class CodeSource(AbstractApiClass):
    """
        Code source for python-based custom feature groups and models

        Args:
            client (ApiClient): An authenticated API Client instance
            sourceType (str): The type of the source, one of TEXT, FILE_UPLOAD, or APPLICATION_CONNECTOR
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
        return f"CodeSource(source_type={repr(self.source_type)},\n  source_code={repr(self.source_code)},\n  application_connector_id={repr(self.application_connector_id)},\n  application_connector_info={repr(self.application_connector_info)},\n  package_requirements={repr(self.package_requirements)},\n  status={repr(self.status)},\n  error={repr(self.error)},\n  publishing_msg={repr(self.publishing_msg)},\n  module_dependencies={repr(self.module_dependencies)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'source_type': self.source_type, 'source_code': self.source_code, 'application_connector_id': self.application_connector_id, 'application_connector_info': self.application_connector_info, 'package_requirements': self.package_requirements, 'status': self.status, 'error': self.error, 'publishing_msg': self.publishing_msg, 'module_dependencies': self.module_dependencies}
