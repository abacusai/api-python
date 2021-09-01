

Abacus.AI Feature Store API 
============


The Abacus.AI platform allows you to process, join and transform raw tabular data present in various data sources (like AWS S3, Snowflake, Redshift, GCS, Salesforce, Marketo) into `features` for the purposes of building machine learning models and running predictions on them. Features are organized into `Feature Groups` and can be easily specified using ANSI-SQL or Python code. Specifying feature groups in Abacus.AI also allows for a clear path to productionizing data pipelines for regular model training, offline predictions and online serving of predictions using a custom or Abacus.AI built model.

## Core concepts


|Concept|Definition  |
|--------|--|
|   Datasets     |A dataset is a named table definition consisting of a data source (an external system connection, a blob storage URI, or a file upload) and a schema (list of column names along with their data types). A dataset version represents actual materialized data created from this definition. Dataset versions are immutable. Datasets can be setup to refresh periodically - which will result in new versions being created automatically from the data source (not applicable for uploads). Every dataset has a table name that is unique to the organization.|
|   Feature Groups     |A feature group is a named table definition which is based on a transformation of the features from datasets or other feature groups. Feature group definitions can be specified using ANSI SQL transformations which reference other dataset and feature group table names directly in the SQL statement. Feature group definitions can also be specified using a user-provided Python function which returns a Pandas Dataframe. Similar to datasets, Feature Groups are just a definition of the transormations and aren't actually applied until you create a Feature Group Version to materialize the data. This can be done via the API or on a refresh schedule. |
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

We'll be using the file connector for the demo purposes as we support reading from publicly accesible buckets, however you can verify your own private buckets on the [Connected Services Page](https://abacus.ai/app/profile/connected_services)

When creating a dataset, you must assign a **Dataset Table Name** which is unique to your organization and used when building derivative Feature Groups. **Dataset Table Names** *must* begin with `datasets_`, if a table name is provided that does not begin with `datasets_`, it will automatically be prepended.

We'll create two datasets, one containing an event log and the other containing item metadata
```python
events_dataset = client.create_dataset_from_file_connector(location='s3://abacusai.exampledatasets/pers_promotion/events.csv', dataset_table_name='datasets_event_log')
items_datasetsclient.create_dataset_from_file_connector(location='s3://abacusai.exampledatasets/pers_promotion/item_categories.csv', dataset_table_name='datasets_item_metadata')
```
Finally, we can create a feature group from these datasets, sepcifying what columns we want as features, and how to join the two tables together. We can do this via ANSI SQL statements or python functions:

### ANSI SQL
```python
feature_group = client.create_feature_group(table_name='joined_events_data', sql='SELECT * FROM datasets_abc JOIN datasets_second USING (id)')
```

### Python Functions

To create a feature group backed by a Python function, we have first provide the source code for the function in a valid python file. In this example, we are using pandas functions in our function. We will run the code in a container which has a Python 3.8 environment with a of standard list of python libraries (specified here).  
````python
fg_code = '''
import pandas as pd
def item_filtering(event_df, items_df):
    final_df = pd.merge(items_df, event_df['item_id'], how='inner', on='item_id')
    final_df = final_df[final_df['timestamp'] < datetime.datetime.now() - datetime.timedelta(days=180)]
    return final_df
'''
feature_group = client.create_feature_group_from_function(table_name='joined_events_data', function_source_code=function_code, input_feature_groups=['datasets_event_log', 'datasets_item_metadata'])
````


##
Finally, we can add descriptive tags to our feature group and add it to our project.
```python
feature_group.add_tag('Joined events log')  # Optional

feature_group.add_to_project(project_id=project.project_id)
```

### Add New Features  [SQL-Based Feature Groups Only] [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HzES-YN4Hzf8dKQuK2STi8uNYkZVtMB0#scrollTo=VT0WqjmfAFdg)
Once you create a feature group, you can add, edit, and remove features by editing the entire sql, or by using utility methods provided:
```python
feature_group.add_feature(name='feature_name', select_expression='CONCAT(col1, col2)')
feature_group.update_feature(name='feature_name', select_expression='CONCAT(col1, col3)')
feature_group.delete_feature(name='feature_name')

feature_group.update_sql(sql='SELECT *, CONCAT(col1, col2) AS feature_name FROM datasets_abc JOIN datasets_second USING (id)')
```
### Create New Feature Groups using Transforms and Joins ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

 - #### SQL feature groups
   ```python
   client.create_feature_group(table_name='complex_sql', select_expression='SELECT id, math, FOO(), BAR() FROM joined_events_data WHERE 1 UNION SELECT * FROM datasets_abc
    ```

### Looking at Materialized Data

````python
df = feature_group.read_latest_version_as_pandas()
````

 - #### Python function feature groups

To create a feature group backed by a Python function, we have first provide the source code for the function in a valid python file. In this example, we are using pandas functions in our function. We will run the code in a container which has a Python 3.8 environment with a of standard list of python libraries (specified here).  

````python

import pandas as pd

def item_filtering(event_df, items_df):
    final_df = pd.merge(items_df, event_df['item_id'], how='inner', on='item_id')
    final_df = final_df[final_df['timestamp'] < datetime.datetime.now() - datetime.timedelta(days=180)]
    return final_df
````

Assuming we have saved this file as `fg_impl.py`, we can use the following snippet to create a python function feature group.
````python
fg_code = open('fg_imp.py').read()
feature_group = client.create_feature_group_from_function(table_name='joined_events_data', function_source_code=fg_code, function_name='item_filtering', input_feature_groups=['datasets_event_log', 'datasets_item_metadata'])
````

We can also use the Abacus.AI client within the python function. The client available during function execution, is a secure version which exposes a read only restricted set of APIs.

````python

import pandas as pd
import abacusai


def construct_fg_from_api():
    client = abacusai.get_client()
    event_fg = client.get_feature_group('datasets_events_log')
    item_fg = client.get_feature_group('datasets_item_metadata')
   
    final_df = pd.merge(items_df, event_df['item_id'], how='inner', on='item_id')
    final_df = final_df[final_df['timestamp'] < datetime.datetime.now() - datetime.timedelta(days=180)]
    return final_df   
````    
    
Assuming we have saved this file as `fg_impl.py`, we can use the following snippet to create a python function feature group.
````python
fg_code = open('fg_imp.py').read()
feature_group = client.create_feature_group_from_function(table_name='joined_events_data', function_source_code=fg_code, function_name='construct_fg_from_api')
````


### Export Materialized Feature Group Data [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HzES-YN4Hzf8dKQuK2STi8uNYkZVtMB0#scrollTo=ompwZJ4nLkqw)

Feature Groups only contain the transformations to apply to the underlying data. In order to apply the transformations, you need to create a **Feature Group Version**.
```python
feature_group_version = feature_group.create_version()
feature_group_version.wait_for_results() # blocks waiting for materialization
```
Now that your data is materialized, we can now export it to a file connector which you have [authorized](https://abacus.ai/app/profile/connected_services) Abacus.AI to be able to write to.

Abacus.AI supports "CSV", "JSON" and "AVRO" as the **Export File Format** of the feature group data.

```python
feature_group_version.export_feature_group_to_file_connector(location='s3://your-bucket/export-location.csv', export_file_format='CSV')
```

### Deploy Feature Groups for Online Featurization of Data [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HzES-YN4Hzf8dKQuK2STi8uNYkZVtMB0#scrollTo=ZleD66xQnCY_)
Feature Groups can be deployed for online data transformations and lookups. Feature Groups with simple join conditions will also support single column id based lookups. The `describeFeatureGroup` method will expose these keys when set. In addition, streaming  

Once set, you can deploy the feature group:
```python
deployment = client.create_feature_group_deployment(project_id=project.project_id, feature_group_id=feature_group.feature_group_id) 
deployment.wait_for_deployment()
deployment_token = client.create_deployment_token(project_id=project.project_id)
```
Now that the deployment is online, you can featurize data by passing in raw dataset rows, a list of lookup keys, or a single lookup key:
```python
client.lookup_features(deployment_id=deployment.deployment_id, deployment_token=deployment_token, query_data={'datasets_event_log': {'@TODO': 'FILL IN DATA'}})
client.lookup_features(deployment_id=deployment.deployment_id, deployment_token=deployment_token, query_data={'user_id': ['id1', 'id2']})
client.lookup_features(deployment_id=deployment.deployment_id, deployment_token=deployment_token, query_data={'item_id': 'item1'})
```

The response will be a list of feature group rows.

### Streaming Data [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HzES-YN4Hzf8dKQuK2STi8uNYkZVtMB0#scrollTo=2IVYnjlvnF5F)

A feature group can be setup to support online updates. This type of feature group is known as a **streaming feature group**. A streaming feature group needs to have a `timestamp` type column that is used as the `recordTimestamp` column. Additionally, a `recordId` column can be specified as the primary key of the feature group, and when that is set, the system will assert that there is only one row for each value of the `recordId` column. When a `recordId` column is specified, the `upsertData` API method is supported, which can be used to partially update data for a specific primary key value. Otherwise, data can be added to a streaming feature group using the `appendData` method. The `recordTimestamp` column is updated to be the time when data is added or updated (and it is not passed in as part of those method calls). 


```python
streaming_feature_group = client.create_streaming_feature_group(table_name='datasets_streaming_interaction_log', record_timestamp_feature='timestamp', [record_id_feature='interaction_id', data_retention_hours=24, data_retention_row_count=1_000_000])
streaming_feature_group.set_streaming_schema(schema=[{'name': 'interaction_id', 'data_type': 'STRING'}, {'name': 'timestamp', 'data_type': 'TIMESTAMP'}, {'name": 'data_column', 'data_type': 'FLOAT'}])
streaming_feature_group.set_streaming_retention_policy(data_retention_hours=48, data_retention_row_count=2_000_000_000)
```

Streaming feature groups can have a retention period which will let the system manage retain only a certain amount of data. This retention policy can be expressed as a period of time or a number of rows. 


To add data to a streaming dataset, we can use the following APIs:
```python
streaming_token = client.create_streaming_token()
```

```python
streaming_feature_group.upsert_data(streaming_token=streaming_token, record_id='user_id_1', data={'data_column': 1}, [record_timestamp=datetime.now() - timedelta(minutes=2)])
client.upsert_data(feature_group_id=streaming_feature_group.feature_group_id, streaming_token=streaming_token, record_id='user_id_1', data={'data_column': 1}, [record_timestamp=datetime.now() - timedelta(minutes=2)])

streaming_feature_group.append_data(streaming_token=streaming_token, data={'data_column': 1})
client.append_data(feature_group_id=streaming_feature_group.feature_group_id, streaming_token=streaming_token, data={'data_column': 1})

```

Another way to manage data in a streaming feature group is to invalidate data before a certain specific timestamp.

```python
streaming_feature_group.invalidate_old_data(valid_after_timestamp=datetime.now() - datetime.timedelta(hours=6))
```

**Concatenating streaming feature group with offline data** Streaming feature groups can be merged with a regular feature group using a **concatenate** operation. Feature groups can be merged if their schema's are compatible and they have the special `recordTimestamp` column and if set, the `recordId` column. The second operand in the concatenate operation will be appended to the first operand (merge target).  

We can specify a `mergeType` option, which can be a `UNION` or an `INTERSECTION`. Depending on this value (defaults to `UNION`), the columns in the final feature group will be a union or an intersection of the two feature groups. 

Concatenation is useful in production settings when we either want to evolve streaming feature groups, or add online updates to a specific table of a feature group that has been developed an initially deployed with offline datasets. 

- If a feature group was developed starting with a streaming feature group and we want to replace past data, we can concatenate data upto a certan point with a new batch data feature group.

```python
streaming_feature_group.concatenate(feature_group_id, merge_type='UNION', afterTimestamp=datetime(2021, 09, 01))
```

- If we started with a batch feature group, built and deployed a final feature group that used this feature group, we can supplement it with realtime data for lookups with a streaming feature group.

```python
feature_group.concatenate(streaming_feature_group_id)
```

If the original feature group was refreshed using a refresh policy, each time the feature group refreshes, we will only add streaming data after the maximum record timestamp of the merge target feature group.



### Open Issues 

- Indexing for streaming lookups
- Constraints in making a FG deployable
