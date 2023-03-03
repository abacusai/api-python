from .return_class import AbstractApiClass


class EdaVersion(AbstractApiClass):
    """
        A version of an eda object

        Args:
            client (ApiClient): An authenticated API Client instance
            edaVersion (str): The unique identifier of a eda version.
            status (str): The current status of the eda object.
            edaId (str): A reference to the eda this version belongs to.
            edaStartedAt (str): The start time and date of the eda process.
            edaCompletedAt (str): The end time and date of the eda process.
            referenceFeatureGroupVersion (list[str]): Feature group version IDs that this refresh pipeline run is analyzing.
            testFeatureGroupVersion (list[str]): Feature group version IDs that this refresh pipeline run is analyzing.
            error (str): Relevant error if the status is FAILED.
    """

    def __init__(self, client, edaVersion=None, status=None, edaId=None, edaStartedAt=None, edaCompletedAt=None, referenceFeatureGroupVersion=None, testFeatureGroupVersion=None, error=None):
        super().__init__(client, edaVersion)
        self.eda_version = edaVersion
        self.status = status
        self.eda_id = edaId
        self.eda_started_at = edaStartedAt
        self.eda_completed_at = edaCompletedAt
        self.reference_feature_group_version = referenceFeatureGroupVersion
        self.test_feature_group_version = testFeatureGroupVersion
        self.error = error

    def __repr__(self):
        return f"EdaVersion(eda_version={repr(self.eda_version)},\n  status={repr(self.status)},\n  eda_id={repr(self.eda_id)},\n  eda_started_at={repr(self.eda_started_at)},\n  eda_completed_at={repr(self.eda_completed_at)},\n  reference_feature_group_version={repr(self.reference_feature_group_version)},\n  test_feature_group_version={repr(self.test_feature_group_version)},\n  error={repr(self.error)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'eda_version': self.eda_version, 'status': self.status, 'eda_id': self.eda_id, 'eda_started_at': self.eda_started_at, 'eda_completed_at': self.eda_completed_at, 'reference_feature_group_version': self.reference_feature_group_version, 'test_feature_group_version': self.test_feature_group_version, 'error': self.error}

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            EdaVersion: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Retrieves a full description of the specified EDA version.

        Args:
            eda_version (str): Unique string identifier of the EDA version.

        Returns:
            EdaVersion: An EDA version.
        """
        return self.client.describe_eda_version(self.eda_version)

    def delete(self):
        """
        Deletes the specified EDA version.

        Args:
            eda_version (str): Unique string identifier of the EDA version to delete.
        """
        return self.client.delete_eda_version(self.eda_version)

    def get_eda_collinearity(self):
        """
        Gets the Collinearity between all features for the Exploratory Data Analysis.

        Args:
            eda_version (str): Unique string identifier associated with the EDA instance.

        Returns:
            EdaCollinearity: An object with a record of correlations between each feature for the EDA.
        """
        return self.client.get_eda_collinearity(self.eda_version)

    def get_eda_data_consistency(self, transformation_feature: str = None):
        """
        Gets the data consistency for the Exploratory Data Analysis.

        Args:
            transformation_feature (str): The transformation feature to get consistency for.

        Returns:
            EdaDataConsistency: Object with duplication, deletion, and transformation data for data consistency analysis for an EDA.
        """
        return self.client.get_eda_data_consistency(self.eda_version, transformation_feature)

    def get_collinearity_for_feature(self, feature_name: str = None):
        """
        Gets the Collinearity for the given feature from the Exploratory Data Analysis.

        Args:
            feature_name (str): Name of the feature for which correlation is shown.

        Returns:
            EdaFeatureCollinearity: Object with a record of correlations for the provided feature for an EDA.
        """
        return self.client.get_collinearity_for_feature(self.eda_version, feature_name)

    def get_feature_association(self, reference_feature_name: str, test_feature_name: str):
        """
        Gets the Feature Association for the given features from the feature group version within the eda_version.

        Args:
            reference_feature_name (str): Name of the feature for feature association (on x-axis for the plots generated for the Feature association in the product).
            test_feature_name (str): Name of the feature for feature association (on y-axis for the plots generated for the Feature association in the product).

        Returns:
            EdaFeatureAssociation: An object with a record of data for the feature association between the two given features for an EDA version.
        """
        return self.client.get_feature_association(self.eda_version, reference_feature_name, test_feature_name)

    def get_eda_forecasting_analysis(self):
        """
        Gets the Forecasting analysis for the Exploratory Data Analysis.

        Args:
            eda_version (str): Unique string identifier associated with the EDA version.

        Returns:
            EdaForecastingAnalysis: Object with forecasting analysis that includes sales_across_time, cummulative_contribution, missing_value_distribution, history_length, num_rows_histogram, product_maturity data.
        """
        return self.client.get_eda_forecasting_analysis(self.eda_version)

    def wait_for_eda(self, timeout=1200):
        """
        A waiting call until eda version is ready.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'MONITORING', 'ANALYZING'}, timeout=timeout)

    def get_status(self):
        """
        Gets the status of the eda version.

        Returns:
            str: A string describing the status of the model monitor version, for e.g., pending, complete, etc.
        """
        return self.describe().status
