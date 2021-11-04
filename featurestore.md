

Abacus.AI Feature Store API
============


The Abacus.AI platform allows you to process, join and transform raw tabular data present in various data sources (like AWS S3, Snowflake, Redshift, GCS, Salesforce, Marketo) into `features` for the purposes of building machine learning models and running predictions on them. Features are organized into `Feature Groups` and can be easily specified using ANSI-SQL or Python code. Specifying feature groups in Abacus.AI also allows for a clear path to productionizing data pipelines for regular model training, offline predictions and online serving of predictions using a custom or Abacus.AI built model.

## Core concepts


|Concept|Definition  |
|--------|--|
|   Datasets     |A dataset is a named table definition consisting of a data source (an external system connection, a blob storage URI, or a file upload) and a schema (list of column names along with their data types). A dataset version represents actual materialized data created from this definition. Dataset versions are immutable. Datasets can be setup to refresh periodically - which will result in new versions being created automatically from the data source (not applicable for uploads). Every dataset has a table name that is unique to the organization.|
|   Feature Groups     |A feature group is a named table definition which is based on a transformation of the features from datasets or other feature groups. Feature group definitions can be specified using ANSI SQL transformations which reference other dataset and feature group table names directly in the SQL statement. Feature group definitions can also be specified using a user-provided Python function which returns a Pandas Dataframe. Similar to datasets, Feature Groups are just a definition of the transformations and aren't actually applied until you create a Feature Group Version to materialize the data. This can be done via the API or on a refresh schedule. |
| Feature | A column in a feature group. |
| Nested Feature Group | A type of Feature Group that supports time-based windowing of data |
| Feature Group Version | A materialized snapshot of a Feature Group's data |
| Project | Entity in Abacus.AI which can contain machine learning models, deployments for online serving of feature groups, and pipelines for training models on feature group data |
| Organization | Entity in Abacus.AI which corresponds to a collection of users who belong to an organization. Datasets, Feature Groups, and Projects are scoped to an organization. Table names across datasets and feature groups are unique to an organization. |

## Setting up a new Feature Store Project [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HzES-YN4Hzf8dKQuK2STi8uNYkZVtMB0#scrollTo=8idfft0im5ci)
To create a new Feature Store project, call the `createProject` API with a name and the **FEATURE_STORE** use case.
```python
project = client.create_project(name='My first Feature Store Project', use_case='FEATURE_STORE')
```

## Create Dataset Definitions and associated Feature Groups [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HzES-YN4Hzf8dKQuK2STi8uNYkZVtMB0#scrollTo=oi6qwR46m71i)

Datasets can be created via uploads [\[example\]](https://github.com/abacusai), file connectors  [\[example\]](https://github.com/abacusai) (blob storage providers such as S3 or GCP Storage), or database connectors  [\[example\]](https://github.com/abacusai) (Salesforce, Snowflake, BigQuery, etc.).

We'll be using the file connector for the demo purposes as we support reading from publicly accessible buckets, however you can verify your own private buckets on the [Connected Services Page](https://abacus.ai/app/profile/connected_services)

When creating a dataset, you must assign a **Feature Group Table Name** which is unique to your organization and used when building derivative Feature Groups. 
We'll create two datasets, one containing an event log and the other containing item metadata
```python
events_dataset = client.create_dataset_from_file_connector(
           name='Events Log', 
           location='s3://abacusai.exampledatasets/pers_promotion/events.csv', 
           table_name='event_log'
)
items_dataset = client.create_dataset_from_file_connector(
           name='Items Data', 
           location='s3://abacusai.exampledatasets/pers_promotion/item_categories.csv', 
           table_name='item_metadata'
)
```
Finally, we can create a feature group from these datasets, specifying what columns we want as features, and how to join the two tables together. We can do this via ANSI SQL statements or python functions:

### ANSI SQL
```python
feature_group = client.create_feature_group(
     table_name='joined_events_data', 
     sql='SELECT * FROM event_log JOIN item_metadata USING (item_id) WHERE timestamp > NOW() - INTERVAL 180 DAY'
)
```

### Python Functions

To create a feature group backed by a Python function, we have first provide the source code for the function in a valid python file. In this example, we are using pandas functions in our function. We will run the code in a container which has a Python 3.8 environment with a of standard list of python libraries. 

````python

import pandas as pd

def item_filtering(items_df, events_df):
    final_df = pd.merge(items_df, events_df['item_id'], how='inner', on='item_id')
    final_df = final_df[final_df['timestamp'] < datetime.datetime.now() - datetime.timedelta(days=180)]
    return final_df
````

Assuming we have saved this file as `fg_impl.py`, we can use the following snippet to create a python function feature group.
````python
fg_code = open('fg_impl.py').read()
feature_group = client.create_feature_group_from_function(table_name='joined_events_data', 
                                                          function_source_code=fg_code, 
                                                          function_name='item_filtering', 
                                                          input_feature_groups=['item_metadata', 'event_log'])
````


### Add New Features  [SQL-Based Feature Groups Only] [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HzES-YN4Hzf8dKQuK2STi8uNYkZVtMB0#scrollTo=VT0WqjmfAFdg)
Once you create a feature group, you can add, edit, and remove features by editing the entire sql, or by using utility methods provided:
```python
feature_group.add_feature(name='feature_name', select_expression='CONCAT(col1, col2)')
feature_group.update_feature(name='feature_name', select_expression='CONCAT(col1, col3)')
feature_group.delete_feature(name='feature_name')

feature_group.update_sql(
       sql='SELECT *, CONCAT(col1, col2) AS feature_name FROM datasets_abc JOIN datasets_second USING (id)'
)
```


### Looking at Materialized Data

````python
df = feature_group.latest_feature_group_version.load_as_pandas()
````

 ### Inputless Python function feature groups

If we want to use a Python function to dynamically read other tables and construct data without having to pre-specify the list of inputs, we can use the Abacus.AI client within the python function. The client available during function execution is a secure version, which exposes a read only restricted set of APIs.

````python

import pandas as pd
import abacusai


def construct_fg_from_api():
    client = abacusai.get_client()
    items_fg = client.get_feature_group('item_metadata')
    events_fg = client.get_feature_group('events_log')
    items_df = items_fg.load_as_pandas()
    events_df = events_fg.load_as_pandas()

    final_df = pd.merge(items_df, events_df['item_id'], how='inner', on='item_id')
    final_df = final_df[final_df['timestamp'] < datetime.datetime.now() - datetime.timedelta(days=180)]
    return final_df
````

For this type of no arguments function, can use the following snippet to create a python function feature group.
````python
feature_group = client.create_feature_group_from_function(
           table_name='joined_events_data', 
           function_source_code=fg_code, 
           function_name='construct_fg_from_api'
)
````

### Point In Time Features

Abacus.AI also supports defining and querying point in time features. Say we want to calculate the number of times a certain event has occurred within a historical window when another event occurred (for e.g, number of views 30 minutes before a purchase), we can associate a historical activity table with another table which records purchases. 

```python
purchases_feature_group.add_point_in_time_feature('num_views_last_30', 
                                                  aggregation_keys=['user_id', 'site_id'], 
                                                  timestamp_key='purchase_timestamp', 
                                                  history_table_name='activity_log', 
                                                  historical_timestamp_key='activity_timestamp', 
                                                  lookback_window_seconds=300, 
                                                  expression='COUNT(1)') 
```

The `add_point_in_time_feature` API method uses the aggregation_key_features to match up the `purchases` and `activity` tables, and for each point in the `purchases` table, retrieves all rows from the `activity` table which have a timestamp within 5 minutes in the past of the purchase timestamp, and evaluates a aggregation expression on those rows. 


A slightly different example shows how to calculate the click through rate from the last 100 events in the activity log.
```python
purchases_feature_group.add_point_in_time_feature(
                     'recent_clicks', 
                     aggregation_keys=['user_id', 'site_id'], 
                     timestamp_key='purchase_timestamp', 
                     history_table_name='activity_log', 
                     historical_timestamp_key='activity_timestamp', 
                     lookback_count=100, 
                     expression='SUM(IF(event_type = "click", 1, 0))'
)
purchases_feature_group.add_point_in_time_feature(
                     'recent_views', 
                     aggregation_keys=['user_id', 'site_id'], 
                     timestamp_key='purchase_timestamp', 
                     history_table_name='activity_log', 
                     historical_timestamp_key='activity_timestamp', 
                     lookback_count=100, 
                     expression='SUM(IF(event_type = "view", 1, 0))'
)
```


### Tags on Feature Groups
To better organize feature groups, we can add descriptive tags to our feature group and add it to our project.
```python
feature_group.add_tag('Joined events log')  # Optional

feature_group.add_to_project(project_id=project.project_id)
```

### Export Materialized Feature Group Data [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HzES-YN4Hzf8dKQuK2STi8uNYkZVtMB0#scrollTo=ompwZJ4nLkqw)

Feature Groups only contain the transformations to apply to the underlying data. In order to apply the transformations, you need to create a **Feature Group Version**.
```python
feature_group_version = feature_group.create_version()
feature_group_version.wait_for_results() # blocks waiting for materialization
```
Now that your data is materialized, we can now export it to a file connector which you have [authorized](https://abacus.ai/app/profile/connected_services) Abacus.AI to be able to write to.

Abacus.AI supports "CSV", "JSON" and "AVRO" as the **Export File Format** of the feature group data.

```python
feature_group_version.export_to_file_connector(location='s3://your-bucket/export-location.csv', export_file_format='CSV')
```

### Deploy Feature Groups for Online Featurization of Data [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HzES-YN4Hzf8dKQuK2STi8uNYkZVtMB0#scrollTo=ZleD66xQnCY_)
Feature Groups can be deployed for online data transformations and lookups. Feature Groups with simple join conditions will also support single column id based lookups. The `describeFeatureGroup` method will expose these keys when set. In addition, streaming

Once set, you can deploy the feature group:
```python
deployment = client.create_deployment(feature_group_id=feature_group.feature_group_id)
deployment.wait_for_deployment()
deployment_token = client.create_deployment_token(project_id=project.project_id).deployment_token
```
Now that the deployment is online, you can featurize data by passing in raw dataset rows, a list of lookup keys, or a single lookup key:
```python
client.lookup_features(deployment_id=deployment.deployment_id, 
                       deployment_token=deployment_token, 
                       query_data={'user_id': ['id1', 'id2']})
client.lookup_features(deployment_id=deployment.deployment_id, 
                       deployment_token=deployment_token, 
                       query_data={'item_id': 'item1'})
```

The response will be a list of feature group rows.

### Streaming Data [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HzES-YN4Hzf8dKQuK2STi8uNYkZVtMB0#scrollTo=2IVYnjlvnF5F)

A feature group project can be setup to support online updates. To accomplish this, we need to configure a **streaming dataset**.

**Streaming feature groups** All streaming datasets (like other datasets) have an associated feature group. You can use this feature group to include streaming data in another project. This feature group can map the `primaryKey`, `updateTimestampKey`, and `lookupKeys` features. We can also explicitly set a schema on this feature group if we want to start configuring it before we actually stream data and infer schema from the streaming dataset. Streaming feature groups need to have a `timestamp` type column that can be used as the `updateTimestampKey` column. Additionally, a `primaryKey` column can be specified as the primary key of the streaming dataset, and when this property is set, there is an implicit assertion that there is only one row for each value of the `primaryKey` column. When a `primaryKey` column is specified, the `upsertData` API method is supported, which can be used to partially update data for a specific primary key value. In addition, streaming data can be indexed by lookup columns Otherwise, data can be added to a streaming dataset using the `appendData` method. The `updateTimestampKey` column is updated to be the time when data is added or updated (and it is not passed in as part of those method calls). To facilitate online look ups, we can mark columns in the streaming feature group as lookup keys.

Streaming datasets can have a retention period which will let the system manage retain only a certain amount of data. This retention policy can be expressed as a period of time or a number of rows.

```python
streaming_dataset_users = client.create_streaming_dataset(table_name='streaming_user_data')
streaming_feature_group_users = client.describe_feature_group_by_table_name(table_name='streaming_user_data')
streaming_feature_group_user.set_indexing_config(update_timestamp_key='update_timestamp', primary_key='user_id')
streaming_dataset_users.set_streaming_retention_policy(retention_hours=48, retention_row_count=2_000_000_000)
```

To add data to a streaming dataset, we can use the following APIs:
```python
streaming_token = client.create_streaming_token().streaming_token
```

```python
streaming_feature_group.upsert_data(streaming_token=streaming_token, 
                                    data={'user_id': 'user_id_1', 
                                          'data_column': 1, 
                                          'update_timestamp': datetime.now() - timedelta(minutes=2)})
client.upsert_data(feature_group_id=streaming_feature_group.feature_group_id, 
                   streaming_token=streaming_token, 
                   data={'user_id': 'user_id_1', 
                         'data_column': 1, 
                         'update_timestamp': datetime.now() - timedelta(minutes=2)})
```


We can also create a streaming feature group which behaves like a log of events with an index.

```python
streaming_dataset_user_activity = client.create_streaming_dataset(table_name='streaming_user_activity')
streaming_feature_group_user_activity = client.describe_feature_group_by_table_name(table_name='streaming_user_activity')
streaming_feature_group_user_activity.set_indexing_config(update_timestamp_key='event_timestamp', lookup_keys=['user_id'])
```

Data can be added to this dataset using the append_data api call. If the `updateTimestampKey` attribute is not set, we use the server receive timestamp as the value for the `updateTimestampKey`

```python
streaming_feature_group_user_activity.append_data(streaming_token=streaming_token, 
                                                  data={'user_id': '1ae2ee', 
                                                        'item_id': '12ef11', 
                                                        'action': 'click', 
                                                        'num_items': 3})
client.append_data(feature_group_id=streaming_feature_group_user_activity.feature_group_id, 
                   streaming_token=streaming_token, 
                   data={'user_id': '1ae2ee', 'item_id': '12ef11', 'action': 'click', 'num_items': 3})

```

Another way to manage data in a streaming dataset is to invalidate data before a certain specific timestamp.

```python
streaming_feature_group_user_activity.invalidate_streaming_data(invalid_before_timestamp=datetime.now() - datetime.timedelta(hours=6))
```

**Concatenating streaming feature group with offline data** Streaming feature groups can be merged with a regular feature group using a **concatenate** operation. Feature groups can be merged if their schema's are compatible and they have the special `updateTimestampKey` column and if set, the `primaryKey` column. The second operand in the concatenate operation will be appended to the first operand (merge target).

We can specify a `mergeType` option, which can be a `UNION` or an `INTERSECTION`. Depending on this value (defaults to `UNION`), the columns in the final feature group will be a union or an intersection of the two feature groups.

Concatenation is useful in production settings when we either want to evolve streaming feature groups, or add online updates to a specific table of a feature group that has been developed an initially deployed with offline datasets.

- If a feature group was developed starting with a streaming feature group and we want to replace past data, we can concatenate data up to a certain point with a new batch data feature group.

```python
streaming_feature_group_user_activity.concatenate_data(feature_group_id, merge_type='UNION', 
                                                       replace_until_timestamp=datetime(2021, 09, 01))
```

- If we started with a batch feature group, built and deployed a final feature group that used this feature group, we can supplement it with realtime data for lookups with a streaming feature group.

```python
feature_group.concatenate_data(streaming_feature_group_user_activity.feature_group_id)
```

If the original feature group was refreshed using a refresh policy, each time the feature group refreshes, we will only add streaming data after the maximum record timestamp of the merge target feature group.
