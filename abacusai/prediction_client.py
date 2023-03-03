import io
from typing import Dict

from .client import BaseApiClient, ClientOptions


class PredictionClient(BaseApiClient):
    """
    Abacus.AI Prediction API Client. Does not utilize authentication and only contains public prediction methods

    Args:
        client_options (ClientOptions): Optional API client configurations
    """

    def __init__(self, client_options: ClientOptions = None):
        super().__init__(api_key=None, client_options=client_options, skip_version_check=True)

    def predict_raw(self, deployment_token: str, deployment_id: str, **kwargs):
        """Raw interface for returning predictions from Plug and Play deployments.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            **kwargs (dict): Arbitrary key/value pairs may be passed in and is sent as part of the request body."""

        return self._call_api('predict', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body=kwargs)

    def lookup_features(self, deployment_token: str, deployment_id: str, query_data: dict = {}, limit_results: int = None, result_columns: list = None) -> Dict:
        """Returns the feature group deployed in the feature store project.

        Args:
            deployment_token (str): A deployment token used to authenticate access to created deployments. This token only authorizes predictions on deployments in this project, so it can be safely embedded inside an application or website.
            deployment_id (str): A unique identifier for a deployment created under the project.
            query_data (dict): A dictionary where the key is the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and the value is the unique value of the same entity.
            limit_results (int): If provided, will limit the number of results to the value specified.
            result_columns (list): If provided, will limit the columns present in each result to the columns specified in this list."""
        return self._call_api('lookupFeatures', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'limitResults': limit_results, 'resultColumns': result_columns}, server_override=self.default_prediction_url)

    def predict(self, deployment_token: str, deployment_id: str, query_data: dict = {}) -> Dict:
        """Returns a prediction for Predictive Modeling

        Args:
            deployment_token (str): A deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, and is safe to embed in an application or website.
            deployment_id (str): A unique identifier for a deployment created under the project.
            query_data (dict): A dictionary where the key is the column name (e.g. a column with name 'user_id' in the dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed, and the value is the unique value of the same entity."""
        return self._call_api('predict', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def predict_multiple(self, deployment_token: str, deployment_id: str, query_data: list = {}) -> Dict:
        """Returns a list of predictions for predictive modeling.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, and is safe to embed in an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            query_data (list): A list of dictionaries, where the 'key' is the column name (e.g. a column with name 'user_id' in the dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed, and the 'value' is the unique value of the same entity."""
        return self._call_api('predictMultiple', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def predict_from_datasets(self, deployment_token: str, deployment_id: str, query_data: dict = {}) -> Dict:
        """Returns a list of predictions for Predictive Modeling.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            query_data (dict): A dictionary where the 'key' is the source dataset name, and the 'value' is a list of records corresponding to the dataset rows."""
        return self._call_api('predictFromDatasets', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def predict_lead(self, deployment_token: str, deployment_id: str, query_data: dict, explain_predictions: bool = False, explainer_type: str = None) -> Dict:
        """Returns the probability of a user being a lead based on their interaction with the service/product and their own attributes (e.g. income, assets, credit score, etc.). Note that the inputs to this method, wherever applicable, should be the column names in the dataset mapped to the column mappings in our system (e.g. column 'user_id' mapped to mapping 'LEAD_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): A dictionary containing user attributes and/or user's interaction data with the product/service (e.g. number of clicks, items in cart, etc.).
            explain_predictions (bool): Will explain predictions for leads
            explainer_type (str): Type of explainer to use for explanations"""
        return self._call_api('predictLead', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'explainPredictions': explain_predictions, 'explainerType': explainer_type}, server_override=self.default_prediction_url)

    def predict_churn(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns the probability of a user to churn out in response to their interactions with the item/product/service. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'churn_result' mapped to mapping 'CHURNED_YN' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where the 'key' will be the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and the 'value' will be the unique value of the same entity."""
        return self._call_api('predictChurn', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def predict_takeover(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns a probability for each class label associated with the types of fraud or a 'yes' or 'no' type label for the possibility of fraud. Note that the inputs to this method, wherever applicable, will be the column names in the dataset mapped to the column mappings in our system (e.g., column 'account_name' mapped to mapping 'ACCOUNT_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): A dictionary containing account activity characteristics (e.g., login id, login duration, login type, IP address, etc.)."""
        return self._call_api('predictTakeover', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def predict_fraud(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns the probability of a transaction performed under a specific account being fraudulent or not. Note that the inputs to this method, wherever applicable, should be the column names in your dataset mapped to the column mappings in our system (e.g. column 'account_number' mapped to the mapping 'ACCOUNT_ID' in our system).

        Args:
            deployment_token (str): A deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique identifier to a deployment created under the project.
            query_data (dict): A dictionary containing transaction attributes (e.g. credit card type, transaction location, transaction amount, etc.)."""
        return self._call_api('predictFraud', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def predict_class(self, deployment_token: str, deployment_id: str, query_data: dict = {}, threshold: float = None, threshold_class: str = None, thresholds: list = None, explain_predictions: bool = False, fixed_features: list = None, nested: str = None, explainer_type: str = None) -> Dict:
        """Returns a classification prediction

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model within an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            query_data (dict): A dictionary where the 'Key' is the column name (e.g. a column with the name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and the 'Value' is the unique value of the same entity.
            threshold (float): A float value that is applied on the popular class label.
            threshold_class (str): The label upon which the threshold is added (binary labels only).
            thresholds (list): Maps labels to thresholds (multi-label classification only). Defaults to F1 optimal threshold if computed for the given class, else uses 0.5.
            explain_predictions (bool): If True, returns the SHAP explanations for all input features.
            fixed_features (list): A set of input features to treat as constant for explanations.
            nested (str): If specified generates prediction delta for each index of the specified nested feature.
            explainer_type (str): The type of explainer to use."""
        return self._call_api('predictClass', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'threshold': threshold, 'thresholdClass': threshold_class, 'thresholds': thresholds, 'explainPredictions': explain_predictions, 'fixedFeatures': fixed_features, 'nested': nested, 'explainerType': explainer_type}, server_override=self.default_prediction_url)

    def predict_target(self, deployment_token: str, deployment_id: str, query_data: dict = {}, explain_predictions: bool = False, fixed_features: list = None, nested: str = None, explainer_type: str = None) -> Dict:
        """Returns a prediction from a classification or regression model. Optionally, includes explanations.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            query_data (dict): A dictionary where the 'key' is the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and the 'value' is the unique value of the same entity.
            explain_predictions (bool): If true, returns the SHAP explanations for all input features.
            fixed_features (list): Set of input features to treat as constant for explanations.
            nested (str): If specified, generates prediction delta for each index of the specified nested feature.
            explainer_type (str): The type of explainer to use."""
        return self._call_api('predictTarget', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'explainPredictions': explain_predictions, 'fixedFeatures': fixed_features, 'nested': nested, 'explainerType': explainer_type}, server_override=self.default_prediction_url)

    def get_anomalies(self, deployment_token: str, deployment_id: str, threshold: float = None, histogram: bool = False) -> io.BytesIO:
        """Returns a list of anomalies from the training dataset.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            threshold (float): The threshold score of what is an anomaly. Valid values are between 0.8 and 0.99.
            histogram (bool): If True, will return a histogram of the distribution of all points."""
        return self._call_api('getAnomalies', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'threshold': threshold, 'histogram': histogram}, server_override=self.default_prediction_url)

    def is_anomaly(self, deployment_token: str, deployment_id: str, query_data: dict = None) -> Dict:
        """Returns a list of anomaly attributes based on login information for a specified account. Note that the inputs to this method, wherever applicable, should be the column names in the dataset mapped to the column mappings in our system (e.g. column 'account_name' mapped to mapping 'ACCOUNT_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): The input data for the prediction."""
        return self._call_api('isAnomaly', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def get_forecast(self, deployment_token: str, deployment_id: str, query_data: dict, future_data: list = None, num_predictions: int = None, prediction_start: str = None, explain_predictions: bool = False, explainer_type: str = None) -> Dict:
        """Returns a list of forecasts for a given entity under the specified project deployment. Note that the inputs to the deployed model will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'holiday_yn' mapped to mapping 'FUTURE' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where 'Key' will be the column name (e.g. a column with name 'store_id' in your dataset) mapped to the column mapping ITEM_ID that uniquely identifies the entity against which forecasting is performed and 'Value' will be the unique value of the same entity.
            future_data (list): This will be a list of values known ahead of time that are relevant for forecasting (e.g. State Holidays, National Holidays, etc.). Each element is a dictionary, where the key and the value both will be of type 'str'. For example future data entered for a Store may be [{"Holiday":"No", "Promo":"Yes", "Date": "2015-07-31 00:00:00"}].
            num_predictions (int): The number of timestamps to predict in the future.
            prediction_start (str): The start date for predictions (e.g., "2015-08-01T00:00:00" as input for mid-night of 2015-08-01).
            explain_predictions (bool): Will explain predictions for forecasting
            explainer_type (str): Type of explainer to use for explanations"""
        return self._call_api('getForecast', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'futureData': future_data, 'numPredictions': num_predictions, 'predictionStart': prediction_start, 'explainPredictions': explain_predictions, 'explainerType': explainer_type}, server_override=self.default_prediction_url)

    def get_k_nearest(self, deployment_token: str, deployment_id: str, vector: list, k: int = None, distance: str = None, include_score: bool = False) -> Dict:
        """Returns the k nearest neighbors for the provided embedding vector.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            vector (list): Input vector to perform the k nearest neighbors with.
            k (int): Overrideable number of items to return.
            distance (str): Specify the distance function to use when finding nearest neighbors.
            include_score (bool): If True, will return the score alongside the resulting embedding value."""
        return self._call_api('getKNearest', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'vector': vector, 'k': k, 'distance': distance, 'includeScore': include_score}, server_override=self.default_prediction_url)

    def get_multiple_k_nearest(self, deployment_token: str, deployment_id: str, queries: list):
        """Returns the k nearest neighbors for the queries provided.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            queries (list): List of mappings of format {"catalogId": "cat0", "vectors": [...], "k": 20, "distance": "euclidean"}. See `getKNearest` for additional information about the supported parameters."""
        return self._call_api('getMultipleKNearest', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queries': queries}, server_override=self.default_prediction_url)

    def get_labels(self, deployment_token: str, deployment_id: str, query_data: dict, threshold: None = None) -> Dict:
        """Returns a list of scored labels from a deployment.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): Dictionary where key is "Content" and value is the text from which entities are to be extracted.
            threshold (None): This argument is deprecated and will be ignored."""
        return self._call_api('getLabels', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'threshold': threshold}, server_override=self.default_prediction_url)

    def get_entities_from_pdf(self, deployment_token: str, deployment_id: str, pdf: io.TextIOBase = None, doc_id: str = None, return_extracted_features: bool = False) -> Dict:
        """Extracts text from the provided PDF and returns a list of recognized labels and their scores.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            pdf (io.TextIOBase): (Optional) The pdf to predict on. One of pdf or docId must be specified.
            doc_id (str): (Optional) The pdf to predict on. One of pdf or docId must be specified.
            return_extracted_features (bool): (Optional) If True, will return all extracted features (e.g. all tokens in a page) from the PDF. Default is False."""
        return self._call_api('getEntitiesFromPDF', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id, 'docId': doc_id, 'returnExtractedFeatures': return_extracted_features}, files={'pdf': pdf}, server_override=self.default_prediction_url)

    def get_recommendations(self, deployment_token: str, deployment_id: str, query_data: dict, num_items: int = 50, page: int = 1, exclude_item_ids: list = None, score_field: str = '', scaling_factors: list = None, restrict_items: list = None, exclude_items: list = None, explore_fraction: float = 0.0) -> Dict:
        """Returns a list of recommendations for a given user under the specified project deployment. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'time' mapped to mapping 'TIMESTAMP' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where 'Key' will be the column name (e.g. a column with name 'user_name' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the user against which recommendations are made and 'Value' will be the unique value of the same item. For example, if you have the column name 'user_name' mapped to the column mapping 'USER_ID', then the query must have the exact same column name (user_name) as key and the name of the user (John Doe) as value.
            num_items (int): The number of items to recommend on one page. By default, it is set to 50 items per page.
            page (int): The page number to be displayed. For example, let's say that the num_items is set to 10 with the total recommendations list size of 50 recommended items, then an input value of 2 in the 'page' variable will display a list of items that rank from 11th to 20th.
            exclude_item_ids (list): [DEPRECATED]
            score_field (str): The relative item scores are returned in a separate field named with the same name as the key (score_field) for this argument.
            scaling_factors (list): It allows you to bias the model towards certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1"], "factor": 1.1}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" in reference to which the model recommendations need to be biased; and the key, "factor" takes the factor by which the item scores are adjusted.  Let's take an example where the input to scaling_factors is [{"column": "VehicleType", "values": ["SUV", "Sedan"], "factor": 1.4}]. After we apply the model to get item probabilities, for every SUV and Sedan in the list, we will multiply the respective probability by 1.1 before sorting. This is particularly useful if there's a type of item that might be less popular but you want to promote it or there's an item that always comes up and you want to demote it.
            restrict_items (list): It allows you to restrict the recommendations to certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1", "value3", ...]}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1", "value3", ...]" to which to restrict the recommendations to. Let's take an example where the input to restrict_items is [{"column": "VehicleType", "values": ["SUV", "Sedan"]}]. This input will restrict the recommendations to SUVs and Sedans. This type of restriction is particularly useful if there's a list of items that you know is of use in some particular scenario and you want to restrict the recommendations only to that list.
            exclude_items (list): It allows you to exclude certain items from the list of recommendations. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1", ...]}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" to exclude from the recommendations. Let's take an example where the input to exclude_items is [{"column": "VehicleType", "values": ["SUV", "Sedan"]}]. The resulting recommendation list will exclude all SUVs and Sedans. This is
            explore_fraction (float): Explore fraction."""
        return self._call_api('getRecommendations', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'numItems': num_items, 'page': page, 'excludeItemIds': exclude_item_ids, 'scoreField': score_field, 'scalingFactors': scaling_factors, 'restrictItems': restrict_items, 'excludeItems': exclude_items, 'exploreFraction': explore_fraction}, server_override=self.default_prediction_url)

    def get_personalized_ranking(self, deployment_token: str, deployment_id: str, query_data: dict, preserve_ranks: list = None, preserve_unknown_items: bool = False, scaling_factors: list = None) -> Dict:
        """Returns a list of items with personalized promotions for a given user under the specified project deployment. Note that the inputs to this method, wherever applicable, should be the column names in the dataset mapped to the column mappings in our system (e.g. column 'item_code' mapped to mapping 'ITEM_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model in an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This should be a dictionary with two key-value pairs. The first pair represents a 'Key' where the column name (e.g. a column with name 'user_id' in the dataset) mapped to the column mapping USER_ID uniquely identifies the user against whom a prediction is made and a 'Value' which is the identifier value for that user. The second pair will have a 'Key' which will be the name of the column name (e.g. movie_name) mapped to ITEM_ID (unique item identifier) and a 'Value' which will be a list of identifiers that uniquely identifies those items.
            preserve_ranks (list): List of dictionaries of format {"column": "col0", "values": ["value0, value1"]}, where the ranks of items in query_data is preserved for all the items in "col0" with values, "value0" and "value1". This option is useful when the desired items are being recommended in the desired order and the ranks for those items need to be kept unchanged during recommendation generation.
            preserve_unknown_items (bool): If true, any items that are unknown to the model, will not be reranked, and the original position in the query will be preserved.
            scaling_factors (list): It allows you to bias the model towards certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1"], "factor": 1.1}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" in reference to which the model recommendations need to be biased; and the key, "factor" takes the factor by which the item scores are adjusted. Let's take an example where the input to scaling_factors is [{"column": "VehicleType", "values": ["SUV", "Sedan"], "factor": 1.4}]. After we apply the model to get item probabilities, for every SUV and Sedan in the list, we will multiply the respective probability by 1.1 before sorting. This is particularly useful if there's a type of item that might be less popular but you want to promote it or there's an item that always comes up and you want to demote it."""
        return self._call_api('getPersonalizedRanking', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'preserveRanks': preserve_ranks, 'preserveUnknownItems': preserve_unknown_items, 'scalingFactors': scaling_factors}, server_override=self.default_prediction_url)

    def get_ranked_items(self, deployment_token: str, deployment_id: str, query_data: dict, preserve_ranks: list = None, preserve_unknown_items: bool = False, scaling_factors: list = None) -> Dict:
        """Returns a list of re-ranked items for a selected user when a list of items is required to be reranked according to the user's preferences. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'item_code' mapped to mapping 'ITEM_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary with two key-value pairs. The first pair represents a 'Key' where the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID uniquely identifies the user against whom a prediction is made and a 'Value' which is the identifier value for that user. The second pair will have a 'Key' which will be the name of the column name (e.g. movie_name) mapped to ITEM_ID (unique item identifier) and a 'Value' which will be a list of identifiers that uniquely identifies those items.
            preserve_ranks (list): List of dictionaries of format {"column": "col0", "values": ["value0, value1"]}, where the ranks of items in query_data is preserved for all the items in "col0" with values, "value0" and "value1". This option is useful when the desired items are being recommended in the desired order and the ranks for those items need to be kept unchanged during recommendation generation.
            preserve_unknown_items (bool): If true, any items that are unknown to the model, will not be reranked, and the original position in the query will be preserved
            scaling_factors (list): It allows you to bias the model towards certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1"], "factor": 1.1}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" in reference to which the model recommendations need to be biased; and the key, "factor" takes the factor by which the item scores are adjusted. Let's take an example where the input to scaling_factors is [{"column": "VehicleType", "values": ["SUV", "Sedan"], "factor": 1.4}]. After we apply the model to get item probabilities, for every SUV and Sedan in the list, we will multiply the respective probability by 1.1 before sorting. This is particularly useful if there is a type of item that might be less popular but you want to promote it or there is an item that always comes up and you want to demote it."""
        return self._call_api('getRankedItems', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'preserveRanks': preserve_ranks, 'preserveUnknownItems': preserve_unknown_items, 'scalingFactors': scaling_factors}, server_override=self.default_prediction_url)

    def get_related_items(self, deployment_token: str, deployment_id: str, query_data: dict, num_items: int = 50, page: int = 1, scaling_factors: list = None, restrict_items: list = None, exclude_items: list = None) -> Dict:
        """Returns a list of related items for a given item under the specified project deployment. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'item_code' mapped to mapping 'ITEM_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where the 'key' will be the column name (e.g. a column with name 'user_name' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the user against which related items are determined and the 'value' will be the unique value of the same item. For example, if you have the column name 'user_name' mapped to the column mapping 'USER_ID', then the query must have the exact same column name (user_name) as key and the name of the user (John Doe) as value.
            num_items (int): The number of items to recommend on one page. By default, it is set to 50 items per page.
            page (int): The page number to be displayed. For example, let's say that the num_items is set to 10 with the total recommendations list size of 50 recommended items, then an input value of 2 in the 'page' variable will display a list of items that rank from 11th to 20th.
            scaling_factors (list): It allows you to bias the model towards certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1"], "factor": 1.1}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" in reference to which the model recommendations need to be biased; and the key, "factor" takes the factor by which the item scores are adjusted.  Let's take an example where the input to scaling_factors is [{"column": "VehicleType", "values": ["SUV", "Sedan"], "factor": 1.4}]. After we apply the model to get item probabilities, for every SUV and Sedan in the list, we will multiply the respective probability by 1.1 before sorting. This is particularly useful if there's a type of item that might be less popular but you want to promote it or there's an item that always comes up and you want to demote it.
            restrict_items (list): It allows you to restrict the recommendations to certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1", "value3", ...]}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1", "value3", ...]" to which to restrict the recommendations to. Let's take an example where the input to restrict_items is [{"column": "VehicleType", "values": ["SUV", "Sedan"]}]. This input will restrict the recommendations to SUVs and Sedans. This type of restriction is particularly useful if there's a list of items that you know is of use in some particular scenario and you want to restrict the recommendations only to that list.
            exclude_items (list): It allows you to exclude certain items from the list of recommendations. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1", ...]}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" to exclude from the recommendations. Let's take an example where the input to exclude_items is [{"column": "VehicleType", "values": ["SUV", "Sedan"]}]. The resulting recommendation list will exclude all SUVs and Sedans. This is particularly useful if there's a list of items that you know is of no use in some particular scenario and you don't want to show those items present in that list."""
        return self._call_api('getRelatedItems', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'numItems': num_items, 'page': page, 'scalingFactors': scaling_factors, 'restrictItems': restrict_items, 'excludeItems': exclude_items}, server_override=self.default_prediction_url)

    def get_chat_response(self, deployment_token: str, deployment_id: str, messages: list, search_results: list = None) -> Dict:
        """Return a chat response which continues the conversation based on the input messages and search results.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            messages (list): A list of chronologically ordered messages, starting with a user message and alternating sources. A message is a dict with attributes:     is_user (bool): Whether the message is from the user.      text (str): The message's text.
            search_results (list): A list of chronologically ordered retrieved search results using the deployment. A retrieved search result is a dict with attributes:     msg_id (int): The corresponding message's index.      result (list): List of NlpSearchPrediction objects."""
        return self._call_api('getChatResponse', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'messages': messages, 'searchResults': search_results}, server_override=self.default_prediction_url)

    def get_search_results(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Return the most relevant search results to the search query from the uploaded documents.

        Args:
            deployment_token (str): A token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it can be securely embedded in an application or website.
            deployment_id (str): A unique identifier of a deployment created under the project.
            query_data (dict): A dictionary where the key is "Content" and the value is the text from which entities are to be extracted."""
        return self._call_api('getSearchResults', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def get_sentiment(self, deployment_token: str, deployment_id: str, document: str) -> Dict:
        """Predicts sentiment on a document

        Args:
            deployment_token (str): A token used to authenticate access to deployments created in this project. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for a deployment created under this project.
            document (str): The document to be analyzed for sentiment."""
        return self._call_api('getSentiment', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'document': document}, server_override=self.default_prediction_url)

    def get_entailment(self, deployment_token: str, deployment_id: str, document: str) -> Dict:
        """Predicts the classification of the document

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            document (str): The document to be classified."""
        return self._call_api('getEntailment', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'document': document}, server_override=self.default_prediction_url)

    def get_classification(self, deployment_token: str, deployment_id: str, document: str) -> Dict:
        """Predicts the classification of the document

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            document (str): The document to be classified."""
        return self._call_api('getClassification', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'document': document}, server_override=self.default_prediction_url)

    def get_summary(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns a JSON of the predicted summary for the given document. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'text' mapped to mapping 'DOCUMENT' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): Raw data dictionary containing the required document data - must have a key 'document' corresponding to a DOCUMENT type text as value."""
        return self._call_api('getSummary', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def predict_language(self, deployment_token: str, deployment_id: str, query_data: str) -> Dict:
        """Predicts the language of the text

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments within this project, making it safe to embed this model in an application or website.
            deployment_id (str): A unique string identifier for a deployment created under the project.
            query_data (str): The input string to detect."""
        return self._call_api('predictLanguage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def get_assignments(self, deployment_token: str, deployment_id: str, query_data: dict, forced_assignments: dict = None) -> Dict:
        """Get all positive assignments that match a query.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it can be safely embedded in an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            query_data (dict): Specifies the set of assignments being requested.
            forced_assignments (dict): Set of assignments to force and resolve before returning query results."""
        return self._call_api('getAssignments', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'forcedAssignments': forced_assignments}, server_override=self.default_prediction_url)

    def check_constraints(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Check for any constraints violated by the overrides.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model within an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            query_data (dict): Assignment overrides to the solution."""
        return self._call_api('checkConstraints', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)

    def predict_with_binary_data(self, deployment_token: str, deployment_id: str, blob: io.TextIOBase, blob_key_name: str = 'blob') -> Dict:
        """Make predictions for a given blob, e.g. image, audio

        Args:
            deployment_token (str): A token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model in an application or website.
            deployment_id (str): A unique identifier to a deployment created under the project.
            blob (io.TextIOBase): The multipart/form-data of the data.
            blob_key_name (str): The key to access the blob data in the model query data."""
        return self._call_api('predictWithBinaryData', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id, 'blobKeyName': blob_key_name}, files={'blob': blob}, server_override=self.default_prediction_url)

    def describe_image(self, deployment_token: str, deployment_id: str, image: io.TextIOBase, categories: list, top_n: int = None) -> Dict:
        """Describe the similarity between an image and a list of categories.

        Args:
            deployment_token (str): Authentication token to access created deployments. This token is only authorized to predict on deployments in the current project, and can be safely embedded in an application or website.
            deployment_id (str): Unique identifier of a deployment created under the project.
            image (io.TextIOBase): Image to describe.
            categories (list): List of candidate categories to compare with the image.
            top_n (int): Return the N most similar categories."""
        return self._call_api('describeImage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id, 'categories': categories, 'topN': top_n}, files={'image': image}, server_override=self.default_prediction_url)

    def transcribe_audio(self, deployment_token: str, deployment_id: str, audio: io.TextIOBase) -> Dict:
        """Transcribe the audio

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to make predictions on deployments in this project, so it can be safely embedded in an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            audio (io.TextIOBase): The audio to transcribe."""
        return self._call_api('transcribeAudio', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, files={'audio': audio}, server_override=self.default_prediction_url)

    def classify_image(self, deployment_token: str, deployment_id: str, image: io.TextIOBase) -> Dict:
        """Classify an image.

        Args:
            deployment_token (str): A deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier to a deployment created under the project.
            image (io.TextIOBase): The binary data of the image to classify."""
        return self._call_api('classifyImage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, files={'image': image}, server_override=self.default_prediction_url)

    def classify_pdf(self, deployment_token: str, deployment_id: str, pdf: io.TextIOBase = None) -> Dict:
        """Returns a classification prediction from a PDF

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model within an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            pdf (io.TextIOBase): (Optional) The pdf to predict on. One of pdf or docId must be specified."""
        return self._call_api('classifyPDF', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, files={'pdf': pdf}, server_override=self.default_prediction_url)

    def get_objects_from_image(self, deployment_token: str, deployment_id: str, image: io.TextIOBase) -> Dict:
        """Classify an image.

        Args:
            deployment_token (str): A deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier to a deployment created under the project.
            image (io.TextIOBase): The binary data of the image to detect objects from."""
        return self._call_api('getObjectsFromImage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, files={'image': image}, server_override=self.default_prediction_url)

    def transfer_style(self, deployment_token: str, deployment_id: str, source_image: io.TextIOBase, style_image: io.TextIOBase) -> Dict:
        """Change the source image to adopt the visual style from the style image.

        Args:
            deployment_token (str): A token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model in an application or website.
            deployment_id (str): A unique identifier to a deployment created under the project.
            source_image (io.TextIOBase): The source image to apply the makeup.
            style_image (io.TextIOBase): The image that has the style as a reference."""
        return self._call_api('transferStyle', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, files={'sourceImage': source_image, 'styleImage': style_image}, server_override=self.default_prediction_url)

    def generate_image(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Generate an image from text prompt.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model within an application or website.
            deployment_id (str): A unique identifier to a deployment created under the project.
            query_data (dict): Specifies the text prompt. For example, {'prompt': 'a cat'}"""
        return self._call_api('generateImage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=self.default_prediction_url)
