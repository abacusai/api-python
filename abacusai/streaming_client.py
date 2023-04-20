
from .client import BaseApiClient, ClientOptions


class StreamingClient(BaseApiClient):
    """
    Abacus.AI Streaming API Client. Does not utilize authentication and only contains public streaming methods

    Args:
        client_options (ClientOptions): Optional API client configurations
    """

    def __init__(self, client_options: ClientOptions = None):
        super().__init__(api_key=None, client_options=client_options, skip_version_check=True)
        if client_options and client_options.server:
            self.prediction_endpoint = client_options.server

    def upsert_item_embeddings(self, streaming_token: str, model_id: str, item_id: str, vector: list, catalog_id: str = None):
        """Upserts an embedding vector for an item id for a model_id.

        Args:
            streaming_token (str): The streaming token for authenticating requests to the model.
            model_id (str): A unique string identifier for the model to upsert item embeddings to.
            item_id (str): The item id for which its embeddings will be upserted.
            vector (list): The embedding vector.
            catalog_id (str): The name of the catalog in the model to update."""
        prediction_url = self._get_streaming_endpoint(
            streaming_token, model_id=model_id)
        return self._call_api('upsertItemEmbeddings', 'POST', query_params={'streamingToken': streaming_token}, body={'modelId': model_id, 'itemId': item_id, 'vector': vector, 'catalogId': catalog_id}, server_override=prediction_url)

    def delete_item_embeddings(self, streaming_token: str, model_id: str, item_ids: list, catalog_id: str = None):
        """Deletes KNN embeddings for a list of item IDs for a given model ID.

        Args:
            streaming_token (str): The streaming token for authenticating requests to the model.
            model_id (str): A unique string identifier for the model from which to delete item embeddings.
            item_ids (list): A list of item IDs whose embeddings will be deleted.
            catalog_id (str): An optional name to specify which catalog in a model to update."""
        prediction_url = self._get_streaming_endpoint(
            streaming_token, model_id=model_id)
        return self._call_api('deleteItemEmbeddings', 'POST', query_params={'streamingToken': streaming_token}, body={'modelId': model_id, 'itemIds': item_ids, 'catalogId': catalog_id}, server_override=prediction_url)

    def upsert_multiple_item_embeddings(self, streaming_token: str, model_id: str, upserts: list, catalog_id: str = None):
        """Upserts a knn embedding for multiple item ids for a model_id.

        Args:
            streaming_token (str): The streaming token for authenticating requests to the model.
            model_id (str): The unique string identifier of the model to upsert item embeddings to.
            upserts (list): A list of dictionaries of the form {'itemId': ..., 'vector': [...]} for each upsert.
            catalog_id (str): Name of the catalog in the model to update."""
        prediction_url = self._get_streaming_endpoint(
            streaming_token, model_id=model_id)
        return self._call_api('upsertMultipleItemEmbeddings', 'POST', query_params={'streamingToken': streaming_token}, body={'modelId': model_id, 'upserts': upserts, 'catalogId': catalog_id}, server_override=prediction_url)

    def upsert_data(self, feature_group_id: str, streaming_token: str, data: dict):
        """Update new data into the feature group for a given lookup key record ID if the record ID is found; otherwise, insert new data into the feature group.

        Args:
            feature_group_id (str): A unique string identifier of the streaming feature group to record data to.
            streaming_token (str): The streaming token for authenticating requests.
            data (dict): The data to record, in JSON format."""
        prediction_url = self._get_streaming_endpoint(
            streaming_token, feature_group_id=feature_group_id)
        return self._call_api('upsertData', 'POST', query_params={'streamingToken': streaming_token}, body={'featureGroupId': feature_group_id, 'data': data}, server_override=prediction_url)

    def append_data(self, feature_group_id: str, streaming_token: str, data: dict):
        """Appends new data into the feature group for a given lookup key recordId.

        Args:
            feature_group_id (str): Unique string identifier for the streaming feature group to record data to.
            streaming_token (str): The streaming token for authenticating requests.
            data (dict): The data to record as a JSON object."""
        prediction_url = self._get_streaming_endpoint(
            streaming_token, feature_group_id=feature_group_id)
        return self._call_api('appendData', 'POST', query_params={'streamingToken': streaming_token}, body={'featureGroupId': feature_group_id, 'data': data}, server_override=prediction_url)

    def upsert_multiple_data(self, feature_group_id: str, streaming_token: str, data: list):
        """Update new data into the feature group for a given lookup key recordId if the recordId is found; otherwise, insert new data into the feature group.

        Args:
            feature_group_id (str): Unique string identifier for the streaming feature group to record data to.
            streaming_token (str): The streaming token for authenticating requests.
            data (list): The data to record, as a list of JSON objects."""
        prediction_url = self._get_streaming_endpoint(
            streaming_token, feature_group_id=feature_group_id)
        return self._call_api('upsertMultipleData', 'POST', query_params={'streamingToken': streaming_token}, body={'featureGroupId': feature_group_id, 'data': data}, server_override=prediction_url)

    def append_multiple_data(self, feature_group_id: str, streaming_token: str, data: list):
        """Appends new data into the feature group for a given lookup key recordId.

        Args:
            feature_group_id (str): Unique string identifier of the streaming feature group to record data to.
            streaming_token (str): Streaming token for authenticating requests.
            data (list): Data to record, as a list of JSON objects."""
        prediction_url = self._get_streaming_endpoint(
            streaming_token, feature_group_id=feature_group_id)
        return self._call_api('appendMultipleData', 'POST', query_params={'streamingToken': streaming_token}, body={'featureGroupId': feature_group_id, 'data': data}, server_override=prediction_url)
