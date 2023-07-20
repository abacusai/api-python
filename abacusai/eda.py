from .eda_version import EdaVersion
from .refresh_schedule import RefreshSchedule
from .return_class import AbstractApiClass


class Eda(AbstractApiClass):
    """
        A exploratory data analysis object

        Args:
            client (ApiClient): An authenticated API Client instance
            edaId (str): The unique identifier of the eda object.
            name (str): The user-friendly name for the eda object.
            createdAt (str): Date and time at which the eda object was created.
            projectId (str): The project this eda object belongs to.
            featureGroupId (str): Feature group ID for which eda analysis is being done.
            referenceFeatureGroupVersion (str): Reference Feature group version for data consistency analysis, will be latest feature group version for collinearity analysis.
            testFeatureGroupVersion (str): Test Feature group version for data consistency analysis, will be latest feature group version for collinearity analysis.
            edaConfigs (dict): Configurations for eda object.
            latestEdaVersion (EdaVersion): The latest eda object version.
            refreshSchedules (RefreshSchedule): List of refresh schedules that indicate when the next model version will be trained.
    """

    def __init__(self, client, edaId=None, name=None, createdAt=None, projectId=None, featureGroupId=None, referenceFeatureGroupVersion=None, testFeatureGroupVersion=None, edaConfigs=None, latestEdaVersion={}, refreshSchedules={}):
        super().__init__(client, edaId)
        self.eda_id = edaId
        self.name = name
        self.created_at = createdAt
        self.project_id = projectId
        self.feature_group_id = featureGroupId
        self.reference_feature_group_version = referenceFeatureGroupVersion
        self.test_feature_group_version = testFeatureGroupVersion
        self.eda_configs = edaConfigs
        self.latest_eda_version = client._build_class(
            EdaVersion, latestEdaVersion)
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)

    def __repr__(self):
        return f"Eda(eda_id={repr(self.eda_id)},\n  name={repr(self.name)},\n  created_at={repr(self.created_at)},\n  project_id={repr(self.project_id)},\n  feature_group_id={repr(self.feature_group_id)},\n  reference_feature_group_version={repr(self.reference_feature_group_version)},\n  test_feature_group_version={repr(self.test_feature_group_version)},\n  eda_configs={repr(self.eda_configs)},\n  latest_eda_version={repr(self.latest_eda_version)},\n  refresh_schedules={repr(self.refresh_schedules)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'eda_id': self.eda_id, 'name': self.name, 'created_at': self.created_at, 'project_id': self.project_id, 'feature_group_id': self.feature_group_id, 'reference_feature_group_version': self.reference_feature_group_version, 'test_feature_group_version': self.test_feature_group_version, 'eda_configs': self.eda_configs, 'latest_eda_version': self._get_attribute_as_dict(self.latest_eda_version), 'refresh_schedules': self._get_attribute_as_dict(self.refresh_schedules)}

    def rerun(self):
        """
        Reruns the specified EDA object.

        Args:
            eda_id (str): Unique string identifier of the EDA object to rerun.

        Returns:
            Eda: The EDA object that is being rerun.
        """
        return self.client.rerun_eda(self.eda_id)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            Eda: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Retrieves a full description of the specified EDA object.

        Args:
            eda_id (str): Unique string identifier associated with the EDA object.

        Returns:
            Eda: Description of the EDA object.
        """
        return self.client.describe_eda(self.eda_id)

    def list_versions(self, limit: int = 100, start_after_version: str = None):
        """
        Retrieves a list of versions for a given EDA object.

        Args:
            limit (int): The maximum length of the list of all EDA versions.
            start_after_version (str): The ID of the version after which the list starts.

        Returns:
            list[EdaVersion]: A list of EDA versions.
        """
        return self.client.list_eda_versions(self.eda_id, limit, start_after_version)

    def rename(self, name: str):
        """
        Renames an EDA

        Args:
            name (str): The new name to apply to the model monitor.
        """
        return self.client.rename_eda(self.eda_id, name)

    def delete(self):
        """
        Deletes the specified EDA and all its versions.

        Args:
            eda_id (str): Unique string identifier of the EDA to delete.
        """
        return self.client.delete_eda(self.eda_id)
