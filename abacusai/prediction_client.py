import io
import json
from typing import Dict, List

from .agent_data_execution_result import AgentDataExecutionResult
from .client import BaseApiClient, ClientOptions
from .document_retriever_lookup_result import DocumentRetrieverLookupResult


class PredictionClient(BaseApiClient):
    """
    Abacus.AI Prediction API Client. Does not utilize authentication and only contains public prediction methods

    Args:
        client_options (ClientOptions): Optional API client configurations
    """

    def __init__(self, client_options: ClientOptions = None):
        super().__init__(api_key=None, client_options=client_options, skip_version_check=True)
        if client_options and client_options.server:
            self.prediction_endpoint = client_options.server

    def predict_raw(self, deployment_token: str, deployment_id: str, **kwargs):
        """Raw interface for returning predictions from Plug and Play deployments.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            **kwargs (dict): Arbitrary key/value pairs may be passed in and is sent as part of the request body."""

        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token)
        return self._call_api('predict', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body=kwargs, server_override=prediction_url)

    def lookup_features(self, deployment_token: str, deployment_id: str, query_data: dict, limit_results: int = None, result_columns: list = None) -> Dict:
        """Returns the feature group deployed in the feature store project.

        Args:
            deployment_token (str): A deployment token used to authenticate access to created deployments. This token only authorizes predictions on deployments in this project, so it can be safely embedded inside an application or website.
            deployment_id (str): A unique identifier for a deployment created under the project.
            query_data (dict): A dictionary where the key is the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and the value is the unique value of the same entity.
            limit_results (int): If provided, will limit the number of results to the value specified.
            result_columns (list): If provided, will limit the columns present in each result to the columns specified in this list."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('lookupFeatures', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'limitResults': limit_results, 'resultColumns': result_columns}, server_override=prediction_url)

    def predict(self, deployment_token: str, deployment_id: str, query_data: dict, **kwargs) -> Dict:
        """Returns a prediction for Predictive Modeling

        Args:
            deployment_token (str): A deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, and is safe to embed in an application or website.
            deployment_id (str): A unique identifier for a deployment created under the project.
            query_data (dict): A dictionary where the key is the column name (e.g. a column with name 'user_id' in the dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed, and the value is the unique value of the same entity."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predict', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id, **kwargs}, body={'queryData': query_data, **kwargs}, server_override=prediction_url)

    def predict_multiple(self, deployment_token: str, deployment_id: str, query_data: list) -> Dict:
        """Returns a list of predictions for predictive modeling.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, and is safe to embed in an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            query_data (list): A list of dictionaries, where the 'key' is the column name (e.g. a column with name 'user_id' in the dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed, and the 'value' is the unique value of the same entity."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predictMultiple', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def predict_from_datasets(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns a list of predictions for Predictive Modeling.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            query_data (dict): A dictionary where the 'key' is the source dataset name, and the 'value' is a list of records corresponding to the dataset rows."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predictFromDatasets', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def predict_lead(self, deployment_token: str, deployment_id: str, query_data: dict, explain_predictions: bool = False, explainer_type: str = None) -> Dict:
        """Returns the probability of a user being a lead based on their interaction with the service/product and their own attributes (e.g. income, assets, credit score, etc.). Note that the inputs to this method, wherever applicable, should be the column names in the dataset mapped to the column mappings in our system (e.g. column 'user_id' mapped to mapping 'LEAD_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): A dictionary containing user attributes and/or user's interaction data with the product/service (e.g. number of clicks, items in cart, etc.).
            explain_predictions (bool): Will explain predictions for leads
            explainer_type (str): Type of explainer to use for explanations"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predictLead', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'explainPredictions': explain_predictions, 'explainerType': explainer_type}, server_override=prediction_url)

    def predict_churn(self, deployment_token: str, deployment_id: str, query_data: dict, explain_predictions: bool = False, explainer_type: str = None) -> Dict:
        """Returns the probability of a user to churn out in response to their interactions with the item/product/service. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'churn_result' mapped to mapping 'CHURNED_YN' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where the 'key' will be the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and the 'value' will be the unique value of the same entity.
            explain_predictions (bool): Will explain predictions for churn
            explainer_type (str): Type of explainer to use for explanations"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predictChurn', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'explainPredictions': explain_predictions, 'explainerType': explainer_type}, server_override=prediction_url)

    def predict_takeover(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns a probability for each class label associated with the types of fraud or a 'yes' or 'no' type label for the possibility of fraud. Note that the inputs to this method, wherever applicable, will be the column names in the dataset mapped to the column mappings in our system (e.g., column 'account_name' mapped to mapping 'ACCOUNT_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): A dictionary containing account activity characteristics (e.g., login id, login duration, login type, IP address, etc.)."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predictTakeover', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def predict_fraud(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns the probability of a transaction performed under a specific account being fraudulent or not. Note that the inputs to this method, wherever applicable, should be the column names in your dataset mapped to the column mappings in our system (e.g. column 'account_number' mapped to the mapping 'ACCOUNT_ID' in our system).

        Args:
            deployment_token (str): A deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique identifier to a deployment created under the project.
            query_data (dict): A dictionary containing transaction attributes (e.g. credit card type, transaction location, transaction amount, etc.)."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predictFraud', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def predict_class(self, deployment_token: str, deployment_id: str, query_data: dict, threshold: float = None, threshold_class: str = None, thresholds: Dict = None, explain_predictions: bool = False, fixed_features: list = None, nested: str = None, explainer_type: str = None) -> Dict:
        """Returns a classification prediction

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model within an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            query_data (dict): A dictionary where the 'Key' is the column name (e.g. a column with the name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and the 'Value' is the unique value of the same entity.
            threshold (float): A float value that is applied on the popular class label.
            threshold_class (str): The label upon which the threshold is added (binary labels only).
            thresholds (Dict): Maps labels to thresholds (multi-label classification only). Defaults to F1 optimal threshold if computed for the given class, else uses 0.5.
            explain_predictions (bool): If True, returns the SHAP explanations for all input features.
            fixed_features (list): A set of input features to treat as constant for explanations - only honored when the explainer type is KERNEL_EXPLAINER
            nested (str): If specified generates prediction delta for each index of the specified nested feature.
            explainer_type (str): The type of explainer to use."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predictClass', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'threshold': threshold, 'thresholdClass': threshold_class, 'thresholds': thresholds, 'explainPredictions': explain_predictions, 'fixedFeatures': fixed_features, 'nested': nested, 'explainerType': explainer_type}, server_override=prediction_url)

    def predict_target(self, deployment_token: str, deployment_id: str, query_data: dict, explain_predictions: bool = False, fixed_features: list = None, nested: str = None, explainer_type: str = None) -> Dict:
        """Returns a prediction from a classification or regression model. Optionally, includes explanations.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            query_data (dict): A dictionary where the 'key' is the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the entity against which a prediction is performed and the 'value' is the unique value of the same entity.
            explain_predictions (bool): If true, returns the SHAP explanations for all input features.
            fixed_features (list): Set of input features to treat as constant for explanations - only honored when the explainer type is KERNEL_EXPLAINER
            nested (str): If specified, generates prediction delta for each index of the specified nested feature.
            explainer_type (str): The type of explainer to use."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predictTarget', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'explainPredictions': explain_predictions, 'fixedFeatures': fixed_features, 'nested': nested, 'explainerType': explainer_type}, server_override=prediction_url)

    def get_anomalies(self, deployment_token: str, deployment_id: str, threshold: float = None, histogram: bool = False) -> io.BytesIO:
        """Returns a list of anomalies from the training dataset.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            threshold (float): The threshold score of what is an anomaly. Valid values are between 0.8 and 0.99.
            histogram (bool): If True, will return a histogram of the distribution of all points."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getAnomalies', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'threshold': threshold, 'histogram': histogram}, server_override=prediction_url)

    def get_timeseries_anomalies(self, deployment_token: str, deployment_id: str, start_timestamp: str = None, end_timestamp: str = None, query_data: dict = None, get_all_item_data: bool = False, series_ids: List = None) -> Dict:
        """Returns a list of anomalous timestamps from the training dataset.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            start_timestamp (str): timestamp from which anomalies have to be detected in the training data
            end_timestamp (str): timestamp to which anomalies have to be detected in the training data
            query_data (dict): additional data on which anomaly detection has to be performed, it can either be a single record or list of records or a json string representing list of records
            get_all_item_data (bool): set this to true if anomaly detection has to be performed on all the data related to input ids
            series_ids (List): list of series ids on which the anomaly detection has to be performed"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getTimeseriesAnomalies', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'startTimestamp': start_timestamp, 'endTimestamp': end_timestamp, 'queryData': query_data, 'getAllItemData': get_all_item_data, 'seriesIds': series_ids}, server_override=prediction_url)

    def is_anomaly(self, deployment_token: str, deployment_id: str, query_data: dict = None) -> Dict:
        """Returns a list of anomaly attributes based on login information for a specified account. Note that the inputs to this method, wherever applicable, should be the column names in the dataset mapped to the column mappings in our system (e.g. column 'account_name' mapped to mapping 'ACCOUNT_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): The input data for the prediction."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('isAnomaly', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def get_event_anomaly_score(self, deployment_token: str, deployment_id: str, query_data: dict = None) -> Dict:
        """Returns an anomaly score for an event.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): The input data for the prediction."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getEventAnomalyScore', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def get_forecast(self, deployment_token: str, deployment_id: str, query_data: dict, future_data: list = None, num_predictions: int = None, prediction_start: str = None, explain_predictions: bool = False, explainer_type: str = None, get_item_data: bool = False) -> Dict:
        """Returns a list of forecasts for a given entity under the specified project deployment. Note that the inputs to the deployed model will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'holiday_yn' mapped to mapping 'FUTURE' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where 'Key' will be the column name (e.g. a column with name 'store_id' in your dataset) mapped to the column mapping ITEM_ID that uniquely identifies the entity against which forecasting is performed and 'Value' will be the unique value of the same entity.
            future_data (list): This will be a list of values known ahead of time that are relevant for forecasting (e.g. State Holidays, National Holidays, etc.). Each element is a dictionary, where the key and the value both will be of type 'str'. For example future data entered for a Store may be [{"Holiday":"No", "Promo":"Yes", "Date": "2015-07-31 00:00:00"}].
            num_predictions (int): The number of timestamps to predict in the future.
            prediction_start (str): The start date for predictions (e.g., "2015-08-01T00:00:00" as input for mid-night of 2015-08-01).
            explain_predictions (bool): Will explain predictions for forecasting
            explainer_type (str): Type of explainer to use for explanations
            get_item_data (bool): Will return the data corresponding to items in query"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getForecast', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'futureData': future_data, 'numPredictions': num_predictions, 'predictionStart': prediction_start, 'explainPredictions': explain_predictions, 'explainerType': explainer_type, 'getItemData': get_item_data}, server_override=prediction_url)

    def get_k_nearest(self, deployment_token: str, deployment_id: str, vector: list, k: int = None, distance: str = None, include_score: bool = False, catalog_id: str = None) -> Dict:
        """Returns the k nearest neighbors for the provided embedding vector.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            vector (list): Input vector to perform the k nearest neighbors with.
            k (int): Overrideable number of items to return.
            distance (str): Specify the distance function to use. Options include “dot“, “cosine“, “euclidean“, and “manhattan“. Default = “dot“
            include_score (bool): If True, will return the score alongside the resulting embedding value.
            catalog_id (str): An optional parameter honored only for embeddings that provide a catalog id"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getKNearest', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'vector': vector, 'k': k, 'distance': distance, 'includeScore': include_score, 'catalogId': catalog_id}, server_override=prediction_url)

    def get_multiple_k_nearest(self, deployment_token: str, deployment_id: str, queries: list):
        """Returns the k nearest neighbors for the queries provided.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            queries (list): List of mappings of format {"catalogId": "cat0", "vectors": [...], "k": 20, "distance": "euclidean"}. See `getKNearest` for additional information about the supported parameters."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getMultipleKNearest', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queries': queries}, server_override=prediction_url)

    def get_labels(self, deployment_token: str, deployment_id: str, query_data: dict, return_extracted_entities: bool = False) -> Dict:
        """Returns a list of scored labels for a document.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): Dictionary where key is "Content" and value is the text from which entities are to be extracted.
            return_extracted_entities (bool): (Optional) If True, will return the extracted entities in simpler format"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getLabels', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'returnExtractedEntities': return_extracted_entities}, server_override=prediction_url)

    def get_entities_from_pdf(self, deployment_token: str, deployment_id: str, pdf: io.TextIOBase = None, doc_id: str = None, return_extracted_features: bool = False, verbose: bool = False, save_extracted_features: bool = None) -> Dict:
        """Extracts text from the provided PDF and returns a list of recognized labels and their scores.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            pdf (io.TextIOBase): (Optional) The pdf to predict on. One of pdf or docId must be specified.
            doc_id (str): (Optional) The pdf to predict on. One of pdf or docId must be specified.
            return_extracted_features (bool): (Optional) If True, will return all extracted features (e.g. all tokens in a page) from the PDF. Default is False.
            verbose (bool): (Optional) If True, will return all the extracted tokens probabilities for all the trained labels. Default is False.
            save_extracted_features (bool): (Optional) If True, will save extracted features (i.e. page tokens) so that they can be fetched using the prediction docId. Default is False."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getEntitiesFromPDF', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={'docId': json.dumps(doc_id) if (doc_id is not None and not isinstance(doc_id, str)) else doc_id, 'returnExtractedFeatures': json.dumps(return_extracted_features) if (return_extracted_features is not None and not isinstance(return_extracted_features, str)) else return_extracted_features, 'verbose': json.dumps(verbose) if (verbose is not None and not isinstance(verbose, str)) else verbose, 'saveExtractedFeatures': json.dumps(save_extracted_features) if (save_extracted_features is not None and not isinstance(save_extracted_features, str)) else save_extracted_features}, files={'pdf': pdf}, server_override=prediction_url)

    def get_recommendations(self, deployment_token: str, deployment_id: str, query_data: dict, num_items: int = None, page: int = None, exclude_item_ids: list = None, score_field: str = None, scaling_factors: list = None, restrict_items: list = None, exclude_items: list = None, explore_fraction: float = None, diversity_attribute_name: str = None, diversity_max_results_per_value: int = None) -> Dict:
        """Returns a list of recommendations for a given user under the specified project deployment. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'time' mapped to mapping 'TIMESTAMP' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary where 'Key' will be the column name (e.g. a column with name 'user_name' in your dataset) mapped to the column mapping USER_ID that uniquely identifies the user against which recommendations are made and 'Value' will be the unique value of the same item. For example, if you have the column name 'user_name' mapped to the column mapping 'USER_ID', then the query must have the exact same column name (user_name) as key and the name of the user (John Doe) as value.
            num_items (int): The number of items to recommend on one page. By default, it is set to 50 items per page.
            page (int): The page number to be displayed. For example, let's say that the num_items is set to 10 with the total recommendations list size of 50 recommended items, then an input value of 2 in the 'page' variable will display a list of items that rank from 11th to 20th.
            score_field (str): The relative item scores are returned in a separate field named with the same name as the key (score_field) for this argument.
            scaling_factors (list): It allows you to bias the model towards certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1"], "factor": 1.1}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" in reference to which the model recommendations need to be biased; and the key, "factor" takes the factor by which the item scores are adjusted.  Let's take an example where the input to scaling_factors is [{"column": "VehicleType", "values": ["SUV", "Sedan"], "factor": 1.4}]. After we apply the model to get item probabilities, for every SUV and Sedan in the list, we will multiply the respective probability by 1.1 before sorting. This is particularly useful if there's a type of item that might be less popular but you want to promote it or there's an item that always comes up and you want to demote it.
            restrict_items (list): It allows you to restrict the recommendations to certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1", "value3", ...]}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1", "value3", ...]" to which to restrict the recommendations to. Let's take an example where the input to restrict_items is [{"column": "VehicleType", "values": ["SUV", "Sedan"]}]. This input will restrict the recommendations to SUVs and Sedans. This type of restriction is particularly useful if there's a list of items that you know is of use in some particular scenario and you want to restrict the recommendations only to that list.
            exclude_items (list): It allows you to exclude certain items from the list of recommendations. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1", ...]}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" to exclude from the recommendations. Let's take an example where the input to exclude_items is [{"column": "VehicleType", "values": ["SUV", "Sedan"]}]. The resulting recommendation list will exclude all SUVs and Sedans. This is
            explore_fraction (float): Explore fraction.
            diversity_attribute_name (str): item attribute column name which is used to ensure diversity of prediction results.
            diversity_max_results_per_value (int): maximum number of results per value of diversity_attribute_name."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getRecommendations', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'numItems': num_items, 'page': page, 'excludeItemIds': exclude_item_ids, 'scoreField': score_field, 'scalingFactors': scaling_factors, 'restrictItems': restrict_items, 'excludeItems': exclude_items, 'exploreFraction': explore_fraction, 'diversityAttributeName': diversity_attribute_name, 'diversityMaxResultsPerValue': diversity_max_results_per_value}, server_override=prediction_url)

    def get_personalized_ranking(self, deployment_token: str, deployment_id: str, query_data: dict, preserve_ranks: list = None, preserve_unknown_items: bool = False, scaling_factors: list = None) -> Dict:
        """Returns a list of items with personalized promotions for a given user under the specified project deployment. Note that the inputs to this method, wherever applicable, should be the column names in the dataset mapped to the column mappings in our system (e.g. column 'item_code' mapped to mapping 'ITEM_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model in an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This should be a dictionary with two key-value pairs. The first pair represents a 'Key' where the column name (e.g. a column with name 'user_id' in the dataset) mapped to the column mapping USER_ID uniquely identifies the user against whom a prediction is made and a 'Value' which is the identifier value for that user. The second pair will have a 'Key' which will be the name of the column name (e.g. movie_name) mapped to ITEM_ID (unique item identifier) and a 'Value' which will be a list of identifiers that uniquely identifies those items.
            preserve_ranks (list): List of dictionaries of format {"column": "col0", "values": ["value0, value1"]}, where the ranks of items in query_data is preserved for all the items in "col0" with values, "value0" and "value1". This option is useful when the desired items are being recommended in the desired order and the ranks for those items need to be kept unchanged during recommendation generation.
            preserve_unknown_items (bool): If true, any items that are unknown to the model, will not be reranked, and the original position in the query will be preserved.
            scaling_factors (list): It allows you to bias the model towards certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1"], "factor": 1.1}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" in reference to which the model recommendations need to be biased; and the key, "factor" takes the factor by which the item scores are adjusted. Let's take an example where the input to scaling_factors is [{"column": "VehicleType", "values": ["SUV", "Sedan"], "factor": 1.4}]. After we apply the model to get item probabilities, for every SUV and Sedan in the list, we will multiply the respective probability by 1.1 before sorting. This is particularly useful if there's a type of item that might be less popular but you want to promote it or there's an item that always comes up and you want to demote it."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getPersonalizedRanking', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'preserveRanks': preserve_ranks, 'preserveUnknownItems': preserve_unknown_items, 'scalingFactors': scaling_factors}, server_override=prediction_url)

    def get_ranked_items(self, deployment_token: str, deployment_id: str, query_data: dict, preserve_ranks: list = None, preserve_unknown_items: bool = False, score_field: str = None, scaling_factors: list = None, diversity_attribute_name: str = None, diversity_max_results_per_value: int = None) -> Dict:
        """Returns a list of re-ranked items for a selected user when a list of items is required to be reranked according to the user's preferences. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'item_code' mapped to mapping 'ITEM_ID' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): This will be a dictionary with two key-value pairs. The first pair represents a 'Key' where the column name (e.g. a column with name 'user_id' in your dataset) mapped to the column mapping USER_ID uniquely identifies the user against whom a prediction is made and a 'Value' which is the identifier value for that user. The second pair will have a 'Key' which will be the name of the column name (e.g. movie_name) mapped to ITEM_ID (unique item identifier) and a 'Value' which will be a list of identifiers that uniquely identifies those items.
            preserve_ranks (list): List of dictionaries of format {"column": "col0", "values": ["value0, value1"]}, where the ranks of items in query_data is preserved for all the items in "col0" with values, "value0" and "value1". This option is useful when the desired items are being recommended in the desired order and the ranks for those items need to be kept unchanged during recommendation generation.
            preserve_unknown_items (bool): If true, any items that are unknown to the model, will not be reranked, and the original position in the query will be preserved
            score_field (str): The relative item scores are returned in a separate field named with the same name as the key (score_field) for this argument.
            scaling_factors (list): It allows you to bias the model towards certain items. The input to this argument is a list of dictionaries where the format of each dictionary is as follows: {"column": "col0", "values": ["value0", "value1"], "factor": 1.1}. The key, "column" takes the name of the column, "col0"; the key, "values" takes the list of items, "["value0", "value1"]" in reference to which the model recommendations need to be biased; and the key, "factor" takes the factor by which the item scores are adjusted. Let's take an example where the input to scaling_factors is [{"column": "VehicleType", "values": ["SUV", "Sedan"], "factor": 1.4}]. After we apply the model to get item probabilities, for every SUV and Sedan in the list, we will multiply the respective probability by 1.1 before sorting. This is particularly useful if there is a type of item that might be less popular but you want to promote it or there is an item that always comes up and you want to demote it.
            diversity_attribute_name (str): item attribute column name which is used to ensure diversity of prediction results.
            diversity_max_results_per_value (int): maximum number of results per value of diversity_attribute_name."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getRankedItems', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'preserveRanks': preserve_ranks, 'preserveUnknownItems': preserve_unknown_items, 'scoreField': score_field, 'scalingFactors': scaling_factors, 'diversityAttributeName': diversity_attribute_name, 'diversityMaxResultsPerValue': diversity_max_results_per_value}, server_override=prediction_url)

    def get_related_items(self, deployment_token: str, deployment_id: str, query_data: dict, num_items: int = None, page: int = None, scaling_factors: list = None, restrict_items: list = None, exclude_items: list = None) -> Dict:
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
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getRelatedItems', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'numItems': num_items, 'page': page, 'scalingFactors': scaling_factors, 'restrictItems': restrict_items, 'excludeItems': exclude_items}, server_override=prediction_url)

    def get_chat_response(self, deployment_token: str, deployment_id: str, messages: list, llm_name: str = None, num_completion_tokens: int = None, system_message: str = None, temperature: float = 0.0, filter_key_values: dict = None, search_score_cutoff: float = None, chat_config: dict = None) -> Dict:
        """Return a chat response which continues the conversation based on the input messages and search results.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            messages (list): A list of chronologically ordered messages, starting with a user message and alternating sources. A message is a dict with attributes:     is_user (bool): Whether the message is from the user.      text (str): The message's text.
            llm_name (str): Name of the specific LLM backend to use to power the chat experience
            num_completion_tokens (int): Default for maximum number of tokens for chat answers
            system_message (str): The generative LLM system message
            temperature (float): The generative LLM temperature
            filter_key_values (dict): A dictionary mapping column names to a list of values to restrict the retrieved search results.
            search_score_cutoff (float): Cutoff for the document retriever score. Matching search results below this score will be ignored.
            chat_config (dict): A dictionary specifying the query chat config override."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getChatResponse', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'messages': messages, 'llmName': llm_name, 'numCompletionTokens': num_completion_tokens, 'systemMessage': system_message, 'temperature': temperature, 'filterKeyValues': filter_key_values, 'searchScoreCutoff': search_score_cutoff, 'chatConfig': chat_config}, server_override=prediction_url)

    def get_chat_response_with_binary_data(self, deployment_token: str, deployment_id: str, messages: list, llm_name: str = None, num_completion_tokens: int = None, system_message: str = None, temperature: float = 0.0, filter_key_values: dict = None, search_score_cutoff: float = None, chat_config: dict = None, attachments: None = None) -> Dict:
        """Return a chat response which continues the conversation based on the input messages and search results.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            messages (list): A list of chronologically ordered messages, starting with a user message and alternating sources. A message is a dict with attributes:     is_user (bool): Whether the message is from the user.      text (str): The message's text.
            llm_name (str): Name of the specific LLM backend to use to power the chat experience
            num_completion_tokens (int): Default for maximum number of tokens for chat answers
            system_message (str): The generative LLM system message
            temperature (float): The generative LLM temperature
            filter_key_values (dict): A dictionary mapping column names to a list of values to restrict the retrieved search results.
            search_score_cutoff (float): Cutoff for the document retriever score. Matching search results below this score will be ignored.
            chat_config (dict): A dictionary specifying the query chat config override.
            attachments (None): A dictionary of binary data to use to answer the queries."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getChatResponseWithBinaryData', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={'messages': json.dumps(messages) if (messages is not None and not isinstance(messages, str)) else messages, 'llmName': json.dumps(llm_name) if (llm_name is not None and not isinstance(llm_name, str)) else llm_name, 'numCompletionTokens': json.dumps(num_completion_tokens) if (num_completion_tokens is not None and not isinstance(num_completion_tokens, str)) else num_completion_tokens, 'systemMessage': json.dumps(system_message) if (system_message is not None and not isinstance(system_message, str)) else system_message, 'temperature': json.dumps(temperature) if (temperature is not None and not isinstance(temperature, str)) else temperature, 'filterKeyValues': json.dumps(filter_key_values) if (filter_key_values is not None and not isinstance(filter_key_values, str)) else filter_key_values, 'searchScoreCutoff': json.dumps(search_score_cutoff) if (search_score_cutoff is not None and not isinstance(search_score_cutoff, str)) else search_score_cutoff, 'chatConfig': json.dumps(chat_config) if (chat_config is not None and not isinstance(chat_config, str)) else chat_config}, files=attachments, server_override=prediction_url)

    def get_conversation_response(self, deployment_id: str, message: str, deployment_token: str, deployment_conversation_id: str = None, external_session_id: str = None, llm_name: str = None, num_completion_tokens: int = None, system_message: str = None, temperature: float = 0.0, filter_key_values: dict = None, search_score_cutoff: float = None, chat_config: dict = None, doc_infos: list = None) -> Dict:
        """Return a conversation response which continues the conversation based on the input message and deployment conversation id (if exists).

        Args:
            deployment_id (str): The unique identifier to a deployment created under the project.
            message (str): A message from the user
            deployment_token (str): A token used to authenticate access to deployments created in this project. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_conversation_id (str): The unique identifier of a deployment conversation to continue. If not specified, a new one will be created.
            external_session_id (str): The user supplied unique identifier of a deployment conversation to continue. If specified, we will use this instead of a internal deployment conversation id.
            llm_name (str): Name of the specific LLM backend to use to power the chat experience
            num_completion_tokens (int): Default for maximum number of tokens for chat answers
            system_message (str): The generative LLM system message
            temperature (float): The generative LLM temperature
            filter_key_values (dict): A dictionary mapping column names to a list of values to restrict the retrived search results.
            search_score_cutoff (float): Cutoff for the document retriever score. Matching search results below this score will be ignored.
            chat_config (dict): A dictionary specifiying the query chat config override.
            doc_infos (list): An optional list of documents use for the conversation. A keyword 'doc_id' is expected to be present in each document for retrieving contents from docstore."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getConversationResponse', 'POST', query_params={'deploymentId': deployment_id, 'deploymentToken': deployment_token}, body={'message': message, 'deploymentConversationId': deployment_conversation_id, 'externalSessionId': external_session_id, 'llmName': llm_name, 'numCompletionTokens': num_completion_tokens, 'systemMessage': system_message, 'temperature': temperature, 'filterKeyValues': filter_key_values, 'searchScoreCutoff': search_score_cutoff, 'chatConfig': chat_config, 'docInfos': doc_infos}, server_override=prediction_url)

    def get_conversation_response_with_binary_data(self, deployment_id: str, deployment_token: str, message: str, deployment_conversation_id: str = None, external_session_id: str = None, llm_name: str = None, num_completion_tokens: int = None, system_message: str = None, temperature: float = 0.0, filter_key_values: dict = None, search_score_cutoff: float = None, chat_config: dict = None, attachments: None = None) -> Dict:
        """Return a conversation response which continues the conversation based on the input message and deployment conversation id (if exists).

        Args:
            deployment_id (str): The unique identifier to a deployment created under the project.
            deployment_token (str): A token used to authenticate access to deployments created in this project. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            message (str): A message from the user
            deployment_conversation_id (str): The unique identifier of a deployment conversation to continue. If not specified, a new one will be created.
            external_session_id (str): The user supplied unique identifier of a deployment conversation to continue. If specified, we will use this instead of a internal deployment conversation id.
            llm_name (str): Name of the specific LLM backend to use to power the chat experience
            num_completion_tokens (int): Default for maximum number of tokens for chat answers
            system_message (str): The generative LLM system message
            temperature (float): The generative LLM temperature
            filter_key_values (dict): A dictionary mapping column names to a list of values to restrict the retrived search results.
            search_score_cutoff (float): Cutoff for the document retriever score. Matching search results below this score will be ignored.
            chat_config (dict): A dictionary specifiying the query chat config override.
            attachments (None): A dictionary of binary data to use to answer the queries."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getConversationResponseWithBinaryData', 'POST', query_params={'deploymentId': deployment_id, 'deploymentToken': deployment_token}, data={'message': json.dumps(message) if (message is not None and not isinstance(message, str)) else message, 'deploymentConversationId': json.dumps(deployment_conversation_id) if (deployment_conversation_id is not None and not isinstance(deployment_conversation_id, str)) else deployment_conversation_id, 'externalSessionId': json.dumps(external_session_id) if (external_session_id is not None and not isinstance(external_session_id, str)) else external_session_id, 'llmName': json.dumps(llm_name) if (llm_name is not None and not isinstance(llm_name, str)) else llm_name, 'numCompletionTokens': json.dumps(num_completion_tokens) if (num_completion_tokens is not None and not isinstance(num_completion_tokens, str)) else num_completion_tokens, 'systemMessage': json.dumps(system_message) if (system_message is not None and not isinstance(system_message, str)) else system_message, 'temperature': json.dumps(temperature) if (temperature is not None and not isinstance(temperature, str)) else temperature, 'filterKeyValues': json.dumps(filter_key_values) if (filter_key_values is not None and not isinstance(filter_key_values, str)) else filter_key_values, 'searchScoreCutoff': json.dumps(search_score_cutoff) if (search_score_cutoff is not None and not isinstance(search_score_cutoff, str)) else search_score_cutoff, 'chatConfig': json.dumps(chat_config) if (chat_config is not None and not isinstance(chat_config, str)) else chat_config}, files=attachments, server_override=prediction_url)

    def get_search_results(self, deployment_token: str, deployment_id: str, query_data: dict, num: int = 15) -> Dict:
        """Return the most relevant search results to the search query from the uploaded documents.

        Args:
            deployment_token (str): A token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it can be securely embedded in an application or website.
            deployment_id (str): A unique identifier of a deployment created under the project.
            query_data (dict): A dictionary where the key is "Content" and the value is the text from which entities are to be extracted.
            num (int): Number of search results to return."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getSearchResults', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'num': num}, server_override=prediction_url)

    def get_sentiment(self, deployment_token: str, deployment_id: str, document: str) -> Dict:
        """Predicts sentiment on a document

        Args:
            deployment_token (str): A token used to authenticate access to deployments created in this project. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for a deployment created under this project.
            document (str): The document to be analyzed for sentiment."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getSentiment', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'document': document}, server_override=prediction_url)

    def get_entailment(self, deployment_token: str, deployment_id: str, document: str) -> Dict:
        """Predicts the classification of the document

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            document (str): The document to be classified."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getEntailment', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'document': document}, server_override=prediction_url)

    def get_classification(self, deployment_token: str, deployment_id: str, document: str) -> Dict:
        """Predicts the classification of the document

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            document (str): The document to be classified."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getClassification', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'document': document}, server_override=prediction_url)

    def get_summary(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Returns a JSON of the predicted summary for the given document. Note that the inputs to this method, wherever applicable, will be the column names in your dataset mapped to the column mappings in our system (e.g. column 'text' mapped to mapping 'DOCUMENT' in our system).

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            query_data (dict): Raw data dictionary containing the required document data - must have a key 'document' corresponding to a DOCUMENT type text as value."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getSummary', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def predict_language(self, deployment_token: str, deployment_id: str, query_data: str) -> Dict:
        """Predicts the language of the text

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments within this project, making it safe to embed this model in an application or website.
            deployment_id (str): A unique string identifier for a deployment created under the project.
            query_data (str): The input string to detect."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predictLanguage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def get_assignments(self, deployment_token: str, deployment_id: str, query_data: dict, forced_assignments: dict = None, solve_time_limit_seconds: float = None, include_all_assignments: bool = False) -> Dict:
        """Get all positive assignments that match a query.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it can be safely embedded in an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            query_data (dict): Specifies the set of assignments being requested. The value for the key can be: 1. A simple scalar value, which is matched exactly 2. A list of values, which matches any element in the list 3. A dictionary with keys lower_in/lower_ex and upper_in/upper_ex, which matches values in an inclusive/exclusive range
            forced_assignments (dict): Set of assignments to force and resolve before returning query results.
            solve_time_limit_seconds (float): Maximum time in seconds to spend solving the query.
            include_all_assignments (bool): If True, will return all assignments, including assignments with value 0. Default is False."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getAssignments', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'forcedAssignments': forced_assignments, 'solveTimeLimitSeconds': solve_time_limit_seconds, 'includeAllAssignments': include_all_assignments}, server_override=prediction_url)

    def get_alternative_assignments(self, deployment_token: str, deployment_id: str, query_data: dict, add_constraints: list = None, solve_time_limit_seconds: float = None, best_alternate_only: bool = False) -> Dict:
        """Get alternative positive assignments for given query. Optimal assignments are ignored and the alternative assignments are returned instead.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it can be safely embedded in an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            query_data (dict): Specifies the set of assignments being requested. The value for the key can be: 1. A simple scalar value, which is matched exactly 2. A list of values, which matches any element in the list 3. A dictionary with keys lower_in/lower_ex and upper_in/upper_ex, which matches values in an inclusive/exclusive range
            add_constraints (list): List of constraints dict to apply to the query. The constraint dict should have the following keys: 1. query (dict): Specifies the set of assignment variables involved in the constraint. The format is same as query_data. 2. operator (str): Constraint operator '=' or '<=' or '>='. 3. constant (int): Constraint RHS constant value. 4. coefficient_column (str): Column in Assignment feature group to be used as coefficient for the assignment variables, optional and defaults to 1
            solve_time_limit_seconds (float): Maximum time in seconds to spend solving the query.
            best_alternate_only (bool): When True only the best alternate will be returned, when False multiple alternates are returned"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getAlternativeAssignments', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'addConstraints': add_constraints, 'solveTimeLimitSeconds': solve_time_limit_seconds, 'bestAlternateOnly': best_alternate_only}, server_override=prediction_url)

    def get_optimization_inputs_from_serialized(self, deployment_token: str, deployment_id: str, query_data: dict = None) -> Dict:
        """Get assignments for given query, with new inputs

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it can be safely embedded in an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            query_data (dict): a dictionary with various key: value pairs corresponding to various updated FGs in the FG tree, which we want to update to compute new top level FGs for online solve. (query data will be dict of names: serialized dataframes)"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getOptimizationInputsFromSerialized', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def get_assignments_online_with_new_serialized_inputs(self, deployment_token: str, deployment_id: str, query_data: dict = None, solve_time_limit_seconds: float = None, optimality_gap_limit: float = None) -> Dict:
        """Get assignments for given query, with new inputs

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it can be safely embedded in an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            query_data (dict): a dictionary with assignment, constraint and constraint_equations_df
            solve_time_limit_seconds (float): Maximum time in seconds to spend solving the query.
            optimality_gap_limit (float): Optimality gap we want to come within, after which we accept the solution as valid. (0 means we only want an optimal solution). it is abs(best_solution_found - best_bound) / abs(best_solution_found)"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getAssignmentsOnlineWithNewSerializedInputs', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data, 'solveTimeLimitSeconds': solve_time_limit_seconds, 'optimalityGapLimit': optimality_gap_limit}, server_override=prediction_url)

    def check_constraints(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Check for any constraints violated by the overrides.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model within an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            query_data (dict): Assignment overrides to the solution."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('checkConstraints', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def predict_with_binary_data(self, deployment_token: str, deployment_id: str, blob: io.TextIOBase) -> Dict:
        """Make predictions for a given blob, e.g. image, audio

        Args:
            deployment_token (str): A token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model in an application or website.
            deployment_id (str): A unique identifier to a deployment created under the project.
            blob (io.TextIOBase): The multipart/form-data of the data."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('predictWithBinaryData', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={}, files={'blob': blob}, server_override=prediction_url)

    def describe_image(self, deployment_token: str, deployment_id: str, image: io.TextIOBase, categories: list, top_n: int = None) -> Dict:
        """Describe the similarity between an image and a list of categories.

        Args:
            deployment_token (str): Authentication token to access created deployments. This token is only authorized to predict on deployments in the current project, and can be safely embedded in an application or website.
            deployment_id (str): Unique identifier of a deployment created under the project.
            image (io.TextIOBase): Image to describe.
            categories (list): List of candidate categories to compare with the image.
            top_n (int): Return the N most similar categories."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('describeImage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={'categories': json.dumps(categories) if (categories is not None and not isinstance(categories, str)) else categories, 'topN': json.dumps(top_n) if (top_n is not None and not isinstance(top_n, str)) else top_n}, files={'image': image}, server_override=prediction_url)

    def get_text_from_document(self, deployment_token: str, deployment_id: str, document: io.TextIOBase = None, adjust_doc_orientation: bool = False, save_predicted_pdf: bool = False, save_extracted_features: bool = False) -> Dict:
        """Generate text from a document

        Args:
            deployment_token (str): Authentication token to access created deployments. This token is only authorized to predict on deployments in the current project, and can be safely embedded in an application or website.
            deployment_id (str): Unique identifier of a deployment created under the project.
            document (io.TextIOBase): Input document which can be an image, pdf, or word document (Some formats might not be supported yet)
            adjust_doc_orientation (bool): (Optional) whether to detect the document page orientation and rotate it if needed.
            save_predicted_pdf (bool): (Optional) If True, will save the predicted pdf bytes so that they can be fetched using the prediction docId. Default is False.
            save_extracted_features (bool): (Optional) If True, will save extracted features (i.e. page tokens) so that they can be fetched using the prediction docId. Default is False."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getTextFromDocument', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={'adjustDocOrientation': json.dumps(adjust_doc_orientation) if (adjust_doc_orientation is not None and not isinstance(adjust_doc_orientation, str)) else adjust_doc_orientation, 'savePredictedPdf': json.dumps(save_predicted_pdf) if (save_predicted_pdf is not None and not isinstance(save_predicted_pdf, str)) else save_predicted_pdf, 'saveExtractedFeatures': json.dumps(save_extracted_features) if (save_extracted_features is not None and not isinstance(save_extracted_features, str)) else save_extracted_features}, files={'document': document}, server_override=prediction_url)

    def transcribe_audio(self, deployment_token: str, deployment_id: str, audio: io.TextIOBase) -> Dict:
        """Transcribe the audio

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to make predictions on deployments in this project, so it can be safely embedded in an application or website.
            deployment_id (str): The unique identifier of a deployment created under the project.
            audio (io.TextIOBase): The audio to transcribe."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('transcribeAudio', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={}, files={'audio': audio}, server_override=prediction_url)

    def classify_image(self, deployment_token: str, deployment_id: str, image: io.TextIOBase = None, doc_id: str = None) -> Dict:
        """Classify an image.

        Args:
            deployment_token (str): A deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier to a deployment created under the project.
            image (io.TextIOBase): The binary data of the image to classify. One of image or doc_id must be specified.
            doc_id (str): The document ID of the image. One of image or doc_id must be specified."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('classifyImage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={'docId': json.dumps(doc_id) if (doc_id is not None and not isinstance(doc_id, str)) else doc_id}, files={'image': image}, server_override=prediction_url)

    def classify_pdf(self, deployment_token: str, deployment_id: str, pdf: io.TextIOBase = None) -> Dict:
        """Returns a classification prediction from a PDF

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model within an application or website.
            deployment_id (str): The unique identifier for a deployment created under the project.
            pdf (io.TextIOBase): (Optional) The pdf to predict on. One of pdf or docId must be specified."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('classifyPDF', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={}, files={'pdf': pdf}, server_override=prediction_url)

    def get_cluster(self, deployment_token: str, deployment_id: str, query_data: dict) -> Dict:
        """Predicts the cluster for given data.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            query_data (dict): A dictionary where each 'key' represents a column name and its corresponding 'value' represents the value of that column. For Timeseries Clustering, the 'key' should be ITEM_ID, and its value should represent a unique item ID that needs clustering."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getCluster', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, server_override=prediction_url)

    def get_objects_from_image(self, deployment_token: str, deployment_id: str, image: io.TextIOBase) -> Dict:
        """Classify an image.

        Args:
            deployment_token (str): A deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier to a deployment created under the project.
            image (io.TextIOBase): The binary data of the image to detect objects from."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getObjectsFromImage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={}, files={'image': image}, server_override=prediction_url)

    def score_image(self, deployment_token: str, deployment_id: str, image: io.TextIOBase) -> Dict:
        """Score on image.

        Args:
            deployment_token (str): A deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier to a deployment created under the project.
            image (io.TextIOBase): The binary data of the image to get the score."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('scoreImage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={}, files={'image': image}, server_override=prediction_url)

    def transfer_style(self, deployment_token: str, deployment_id: str, source_image: io.TextIOBase, style_image: io.TextIOBase) -> io.BytesIO:
        """Change the source image to adopt the visual style from the style image.

        Args:
            deployment_token (str): A token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model in an application or website.
            deployment_id (str): A unique identifier to a deployment created under the project.
            source_image (io.TextIOBase): The source image to apply the makeup.
            style_image (io.TextIOBase): The image that has the style as a reference."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('transferStyle', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={}, files={'sourceImage': source_image, 'styleImage': style_image}, streamable_response=True, server_override=prediction_url)

    def generate_image(self, deployment_token: str, deployment_id: str, query_data: dict) -> io.BytesIO:
        """Generate an image from text prompt.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model within an application or website.
            deployment_id (str): A unique identifier to a deployment created under the project.
            query_data (dict): Specifies the text prompt. For example, {'prompt': 'a cat'}"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('generateImage', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'queryData': query_data}, streamable_response=True, server_override=prediction_url)

    def execute_agent(self, deployment_token: str, deployment_id: str, arguments: list = None, keyword_arguments: dict = None) -> Dict:
        """Executes a deployed AI agent function using the arguments as keyword arguments to the agent execute function.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            arguments (list): Positional arguments to the agent execute function.
            keyword_arguments (dict): A dictionary where each 'key' represents the paramter name and its corresponding 'value' represents the value of that parameter for the agent execute function."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('executeAgent', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'arguments': arguments, 'keywordArguments': keyword_arguments}, server_override=prediction_url, timeout=1500)

    def get_matrix_agent_schema(self, deployment_token: str, deployment_id: str, query: str, doc_infos: list = None, deployment_conversation_id: str = None, external_session_id: str = None) -> Dict:
        """Executes a deployed AI agent function using the arguments as keyword arguments to the agent execute function.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            query (str): User input query to initialize the matrix computation.
            doc_infos (list): An optional list of documents use for constructing the matrix. A keyword 'doc_id' is expected to be present in each document for retrieving contents from docstore.
            deployment_conversation_id (str): A unique string identifier for the deployment conversation used for the conversation.
            external_session_id (str): A unique string identifier for the session used for the conversation. If both deployment_conversation_id and external_session_id are not provided, a new session will be created."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getMatrixAgentSchema', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'query': query, 'docInfos': doc_infos, 'deploymentConversationId': deployment_conversation_id, 'externalSessionId': external_session_id}, server_override=prediction_url, timeout=1500)

    def execute_conversation_agent(self, deployment_token: str, deployment_id: str, arguments: list = None, keyword_arguments: dict = None, deployment_conversation_id: str = None, external_session_id: str = None, regenerate: bool = False, doc_infos: list = None, agent_workflow_node_id: str = None) -> Dict:
        """Executes a deployed AI agent function using the arguments as keyword arguments to the agent execute function.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            arguments (list): Positional arguments to the agent execute function.
            keyword_arguments (dict): A dictionary where each 'key' represents the paramter name and its corresponding 'value' represents the value of that parameter for the agent execute function.
            deployment_conversation_id (str): A unique string identifier for the deployment conversation used for the conversation.
            external_session_id (str): A unique string identifier for the session used for the conversation. If both deployment_conversation_id and external_session_id are not provided, a new session will be created.
            regenerate (bool): If True, will regenerate the response from the last query.
            doc_infos (list): An optional list of documents use for the conversation. A keyword 'doc_id' is expected to be present in each document for retrieving contents from docstore.
            agent_workflow_node_id (str): An optional agent workflow node id to trigger agent execution from an intermediate node."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('executeConversationAgent', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'arguments': arguments, 'keywordArguments': keyword_arguments, 'deploymentConversationId': deployment_conversation_id, 'externalSessionId': external_session_id, 'regenerate': regenerate, 'docInfos': doc_infos, 'agentWorkflowNodeId': agent_workflow_node_id}, server_override=prediction_url)

    def lookup_matches(self, deployment_token: str, deployment_id: str, data: str = None, filters: dict = None, num: int = None, result_columns: list = None, max_words: int = None, num_retrieval_margin_words: int = None, max_words_per_chunk: int = None, score_multiplier_column: str = None, min_score: float = None, required_phrases: list = None, filter_clause: str = None, crowding_limits: dict = None, include_text_search: bool = False) -> List[DocumentRetrieverLookupResult]:
        """Lookup document retrievers and return the matching documents from the document retriever deployed with given query.

        Original documents are splitted into chunks and stored in the document retriever. This lookup function will return the relevant chunks
        from the document retriever. The returned chunks could be expanded to include more words from the original documents and merged if they
        are overlapping, and permitted by the settings provided. The returned chunks are sorted by relevance.


        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments within this project, making it safe to embed this model in an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            data (str): The query to search for.
            filters (dict): A dictionary mapping column names to a list of values to restrict the retrieved search results.
            num (int): If provided, will limit the number of results to the value specified.
            result_columns (list): If provided, will limit the column properties present in each result to those specified in this list.
            max_words (int): If provided, will limit the total number of words in the results to the value specified.
            num_retrieval_margin_words (int): If provided, will add this number of words from left and right of the returned chunks.
            max_words_per_chunk (int): If provided, will limit the number of words in each chunk to the value specified. If the value provided is smaller than the actual size of chunk on disk, which is determined during document retriever creation, the actual size of chunk will be used. I.e, chunks looked up from document retrievers will not be split into smaller chunks during lookup due to this setting.
            score_multiplier_column (str): If provided, will use the values in this column to modify the relevance score of the returned chunks. Values in this column must be numeric.
            min_score (float): If provided, will filter out the results with score less than the value specified.
            required_phrases (list): If provided, each result will contain at least one of the phrases in the given list. The matching is whitespace and case insensitive.
            filter_clause (str): If provided, filter the results of the query using this sql where clause.
            crowding_limits (dict): A dictionary mapping metadata columns to the maximum number of results per unique value of the column. This is used to ensure diversity of metadata attribute values in the results. If a particular attribute value has already reached its maximum count, further results with that same attribute value will be excluded from the final result set. An entry in the map can also be a map specifying the limit per attribute value rather than a single limit for all values. This allows a per value limit for attributes. If an attribute value is not present in the map its limit defaults to zero.
            include_text_search (bool): If true, combine the ranking of results from a BM25 text search over the documents with the vector search using reciprocal rank fusion. It leverages both lexical and semantic matching for better overall results. It's particularly valuable in professional, technical, or specialized fields where both precision in terminology and understanding of context are important.

        Returns:
            list[DocumentRetrieverLookupResult]: The relevant documentation results found from the document retriever."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('lookupMatches', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'data': data, 'filters': filters, 'num': num, 'resultColumns': result_columns, 'maxWords': max_words, 'numRetrievalMarginWords': num_retrieval_margin_words, 'maxWordsPerChunk': max_words_per_chunk, 'scoreMultiplierColumn': score_multiplier_column, 'minScore': min_score, 'requiredPhrases': required_phrases, 'filterClause': filter_clause, 'crowdingLimits': crowding_limits, 'includeTextSearch': include_text_search}, parse_type=DocumentRetrieverLookupResult, server_override=prediction_url)

    def get_completion(self, deployment_token: str, deployment_id: str, prompt: str) -> Dict:
        """Returns the finetuned LLM generated completion of the prompt.

        Args:
            deployment_token (str): The deployment token to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): The unique identifier to a deployment created under the project.
            prompt (str): The prompt given to the finetuned LLM to generate the completion."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('getCompletion', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'prompt': prompt}, server_override=prediction_url)

    def execute_agent_with_binary_data(self, deployment_token: str, deployment_id: str, arguments: list = None, keyword_arguments: dict = None, deployment_conversation_id: str = None, external_session_id: str = None, blobs: None = None) -> Dict:
        """Executes a deployed AI agent function with binary data as inputs.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            arguments (list): Positional arguments to the agent execute function.
            keyword_arguments (dict): A dictionary where each 'key' represents the parameter name and its corresponding 'value' represents the value of that parameter for the agent execute function.
            deployment_conversation_id (str): A unique string identifier for the deployment conversation used for the conversation.
            external_session_id (str): A unique string identifier for the session used for the conversation. If both deployment_conversation_id and external_session_id are not provided, a new session will be created.
            blobs (None): A dictionary of binary data to use as inputs to the agent execute function.

        Returns:
            AgentDataExecutionResult: The result of the agent execution"""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('executeAgentWithBinaryData', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, data={'arguments': json.dumps(arguments) if (arguments is not None and not isinstance(arguments, str)) else arguments, 'keywordArguments': json.dumps(keyword_arguments) if (keyword_arguments is not None and not isinstance(keyword_arguments, str)) else keyword_arguments, 'deploymentConversationId': json.dumps(deployment_conversation_id) if (deployment_conversation_id is not None and not isinstance(deployment_conversation_id, str)) else deployment_conversation_id, 'externalSessionId': json.dumps(external_session_id) if (external_session_id is not None and not isinstance(external_session_id, str)) else external_session_id}, parse_type=AgentDataExecutionResult, files=blobs, server_override=prediction_url, timeout=1500)

    def start_autonomous_agent(self, deployment_token: str, deployment_id: str, arguments: list = None, keyword_arguments: dict = None, save_conversations: bool = True) -> Dict:
        """Starts a deployed Autonomous agent associated with the given deployment_conversation_id using the arguments and keyword arguments as inputs for execute function of trigger node.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, making it safe to embed this model in an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            arguments (list): Positional arguments to the agent execute function.
            keyword_arguments (dict): A dictionary where each 'key' represents the parameter name and its corresponding 'value' represents the value of that parameter for the agent execute function.
            save_conversations (bool): If true then a new conversation will be created for every run of the workflow associated with the agent."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('startAutonomousAgent', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'arguments': arguments, 'keywordArguments': keyword_arguments, 'saveConversations': save_conversations}, server_override=prediction_url, timeout=1500)

    def pause_autonomous_agent(self, deployment_token: str, deployment_id: str, deployment_conversation_id: str) -> Dict:
        """Pauses a deployed Autonomous agent associated with the given deployment_conversation_id.

        Args:
            deployment_token (str): The deployment token used to authenticate access to created deployments. This token is only authorized to predict on deployments in this project, making it safe to embed this model in an application or website.
            deployment_id (str): A unique string identifier for the deployment created under the project.
            deployment_conversation_id (str): A unique string identifier for the deployment conversation used for the conversation."""
        prediction_url = self._get_prediction_endpoint(
            deployment_id, deployment_token) if deployment_token else None
        return self._call_api('pauseAutonomousAgent', 'POST', query_params={'deploymentToken': deployment_token, 'deploymentId': deployment_id}, body={'deploymentConversationId': deployment_conversation_id}, server_override=prediction_url, timeout=1500)
