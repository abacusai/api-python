from .return_class import AbstractApiClass


class ResolvedFeatureGroupTemplate(AbstractApiClass):
    """
        Results from resolving a feature group template.

        Args:
            client (ApiClient): An authenticated API Client instance
            featureGroupTemplateId (str): The unique identifier for this feature group template.
            resolvedBindings (dict): A map from template variable names to parameters that were available during template resolution.
            resolvedSql (str): The SQL resulting from resolving the sql template by applying the resolved bindings.
            templateSql (str): SQL that can include variables to be replaced by values from the template config to resolve this template SQL into a valid SQL query for a feature group.
    """

    def __init__(self, client, featureGroupTemplateId=None, resolvedBindings=None, resolvedSql=None, templateSql=None):
        super().__init__(client, None)
        self.feature_group_template_id = featureGroupTemplateId
        self.resolved_bindings = resolvedBindings
        self.resolved_sql = resolvedSql
        self.template_sql = templateSql

    def __repr__(self):
        return f"ResolvedFeatureGroupTemplate(feature_group_template_id={repr(self.feature_group_template_id)},\n  resolved_bindings={repr(self.resolved_bindings)},\n  resolved_sql={repr(self.resolved_sql)},\n  template_sql={repr(self.template_sql)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'feature_group_template_id': self.feature_group_template_id, 'resolved_bindings': self.resolved_bindings, 'resolved_sql': self.resolved_sql, 'template_sql': self.template_sql}
