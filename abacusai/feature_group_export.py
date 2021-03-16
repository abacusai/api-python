

class FeatureGroupExport():
    '''

    '''

    def __init__(self, client, exportId=None, featureGroupId=None, outputLocation=None, status=None, createdAt=None, exportCompletedAt=None):
        self.client = client
        self.id = None
        self.export_id = exportId
        self.feature_group_id = featureGroupId
        self.output_location = outputLocation
        self.status = status
        self.created_at = createdAt
        self.export_completed_at = exportCompletedAt

    def __repr__(self):
        return f"FeatureGroupExport(export_id={repr(self.export_id)}, feature_group_id={repr(self.feature_group_id)}, output_location={repr(self.output_location)}, status={repr(self.status)}, created_at={repr(self.created_at)}, export_completed_at={repr(self.export_completed_at)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'export_id': self.export_id, 'feature_group_id': self.feature_group_id, 'output_location': self.output_location, 'status': self.status, 'created_at': self.created_at, 'export_completed_at': self.export_completed_at}

    def describe(self):
        return self.client.describe_export(self.export_id)

    def wait_for_results(self, timeout=3600):
        return self.client._poll(self, {'PENDING', 'EXPORTING'}, timeout=timeout)

    def get_status(self):
        return self.describe().status

    def get_results(self):
        return self.client.get_export_result(self.export_id)
