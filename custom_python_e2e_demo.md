## How-to guide for custom Python data transforms and models
This notebook provides you with a hands on environment to build and deploy custom python models in the Abacus.AI environment. Custom here refers to both the data transformations required to build a model and the training process used to construct a model from data. Having the custom logic hosted in Abacus.AI then allows for the process to be run automatically to refresh the model as new data arrives and to host the model for the generation of online or batch predictions. In addition, it allows for additional features like monitoring input drift, model performance and other ML Ops requirements.

1. Install the Abacus.AI library.


```python
!pip install abacusai
```

2. Add your Abacus.AI [API Key](https://abacus.ai/app/profile/apikey) generated using the API dashboard as follows:


```python
#@title Abacus.AI API Key

api_key = 'cf45d2********fa79101f7b'  #@param {type: "string"}
```

3. Import the Abacus.AI library and instantiate a client.


```python
from abacusai import ApiClient, ApiException
client = ApiClient(api_key)
```

## 1. Create a Project

In this notebook, we're going to see how to use python to customize models in Abacus. We will cover custom data transforms, model training and prediction handling. Projects that will be hosting a custom model needed to be created with the `PYTHON_MODEL` use case. Note that custom python data transforms can be used in any kind of project and like any other feature group can be shared across projects. However, custom training algorithms and prediction functions are enabled by this use case.


```python
project = client.create_project(name='Demo Python Model', use_case='PYTHON_MODEL')
```

## 2. Creating Datasets

Abacus.AI can read datasets directly from File blob storage

We are going to use a single dataset for this project.
- [Concrete Strength](https://s3.amazonaws.com/abacusai.exampledatasets/predicting/concrete_measurements.csv)


### Add the datasets to Abacus.AI


Using the Create Dataset API, we can tell Abacus.AI the public S3 URI of where to find the datasets.




```python
# if the dataset already exists, skip creation
try: 
  concrete_dataset = client.describe_dataset(client.describe_feature_group_by_table_name('concrete_strength').dataset_id)
except ApiException: # dataset not found
  concrete_dataset = client.create_dataset_from_file_connector(
      name='Concrete Strength',
      table_name='concrete_strength',
      location='s3://abacusai.exampledatasets/predicting/concrete_measurements.csv')
  concrete_dataset.wait_for_inspection()
```

### Load the dataset so we can build and test the transform.

Most of the time it is easiest to develop custom transformations on your local machine. It makes iteration, inspection and debugging easier and often you can do it directly in a notebook environment. To enable simple local development you can use the Abacus.AI client to load your dataset as a pandas dataframe. This tends to work well if your dataset is under `100MB` but for datasets that get much larger you will likely want to construct a sampled feature group for development.

Here we are working with a fairly small dataset so can easily load it into memory. The first block fetches the feature group corresponding to the dataset (datasets are used to move data into Abacus.AI, feature groups are used to consume data for various operations). It initiates a materialization of the feature group to generate a snapshot, waits for it to be ready and then loads it as a pandas dataframe.


```python
concrete_feature_group = concrete_dataset.describe_feature_group()
if not concrete_feature_group.list_versions():
  concrete_feature_group.create_version()
concrete_feature_group.wait_for_materialization()

concrete_df = concrete_feature_group.load_as_pandas()
concrete_df[:10]
```




<div>
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cement</th>
      <th>slag</th>
      <th>flyash</th>
      <th>water</th>
      <th>superplasticizer</th>
      <th>coarseaggregate</th>
      <th>fineaggregate</th>
      <th>age</th>
      <th>csMPa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>540.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1040.000000</td>
      <td>676.0</td>
      <td>28.0</td>
      <td>79.989998</td>
    </tr>
    <tr>
      <th>1</th>
      <td>540.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1055.000000</td>
      <td>676.0</td>
      <td>28.0</td>
      <td>61.889999</td>
    </tr>
    <tr>
      <th>2</th>
      <td>332.500000</td>
      <td>142.500000</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.000000</td>
      <td>594.0</td>
      <td>270.0</td>
      <td>40.270000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>332.500000</td>
      <td>142.500000</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.000000</td>
      <td>594.0</td>
      <td>365.0</td>
      <td>41.049999</td>
    </tr>
    <tr>
      <th>4</th>
      <td>198.600006</td>
      <td>132.399994</td>
      <td>0.0</td>
      <td>192.0</td>
      <td>0.0</td>
      <td>978.400024</td>
      <td>825.5</td>
      <td>360.0</td>
      <td>44.299999</td>
    </tr>
    <tr>
      <th>5</th>
      <td>266.000000</td>
      <td>114.000000</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.000000</td>
      <td>670.0</td>
      <td>90.0</td>
      <td>47.029999</td>
    </tr>
    <tr>
      <th>6</th>
      <td>380.000000</td>
      <td>95.000000</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.000000</td>
      <td>594.0</td>
      <td>365.0</td>
      <td>43.700001</td>
    </tr>
    <tr>
      <th>7</th>
      <td>380.000000</td>
      <td>95.000000</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.000000</td>
      <td>594.0</td>
      <td>28.0</td>
      <td>36.450001</td>
    </tr>
    <tr>
      <th>8</th>
      <td>266.000000</td>
      <td>114.000000</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.000000</td>
      <td>670.0</td>
      <td>28.0</td>
      <td>45.849998</td>
    </tr>
    <tr>
      <th>9</th>
      <td>475.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.000000</td>
      <td>594.0</td>
      <td>28.0</td>
      <td>39.290001</td>
    </tr>
  </tbody>
</table>
</div>



#### Custom Data Transform
We are going to transform the dataset so that flyash is no longer a feature but instead all the other values are transformed according to whether they have `flyash > 0` or not.

The example is not entirely realistic and it is certainly feasible to achieve the same result using SQL. However, the point is to illustrate that you are free to transform the dataset using the full functionality of python and its data frameworks. Here we are using pandas but you can use a wide range of standard python libraries to manipulate the data. Additionally, you can bundle resources with your code, for example small maps or tables, that can be accessed by your function to implement the transform.

Note that we test the function locally by running it against the dataframe loaded from the feature group.


```python
def separate_by_flyash(concrete_strength):
  import pandas as pd
  feature_df = concrete_strength.drop(['flyash'], axis=1)
  no_flyash = feature_df[concrete_strength.flyash == 0.0]
  flyash = feature_df[concrete_strength.flyash > 0.0]
  return pd.concat([no_flyash - no_flyash.assign(age=0).mean(), flyash - flyash.assign(age=0).mean()])

concrete_by_flyash_df = separate_by_flyash(concrete_df)
concrete_by_flyash_df[:10]
```




<div>
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cement</th>
      <th>slag</th>
      <th>water</th>
      <th>superplasticizer</th>
      <th>coarseaggregate</th>
      <th>fineaggregate</th>
      <th>age</th>
      <th>csMPa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>225.962191</td>
      <td>-100.110247</td>
      <td>-24.616784</td>
      <td>-1.555654</td>
      <td>66.642580</td>
      <td>-88.853001</td>
      <td>28.0</td>
      <td>43.218213</td>
    </tr>
    <tr>
      <th>1</th>
      <td>225.962191</td>
      <td>-100.110247</td>
      <td>-24.616784</td>
      <td>-1.555654</td>
      <td>81.642580</td>
      <td>-88.853001</td>
      <td>28.0</td>
      <td>25.118215</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.462191</td>
      <td>42.389753</td>
      <td>41.383216</td>
      <td>-4.055654</td>
      <td>-41.357420</td>
      <td>-170.853001</td>
      <td>270.0</td>
      <td>3.498216</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18.462191</td>
      <td>42.389753</td>
      <td>41.383216</td>
      <td>-4.055654</td>
      <td>-41.357420</td>
      <td>-170.853001</td>
      <td>365.0</td>
      <td>4.278215</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-115.437803</td>
      <td>32.289747</td>
      <td>5.383216</td>
      <td>-4.055654</td>
      <td>5.042604</td>
      <td>60.646999</td>
      <td>360.0</td>
      <td>7.528215</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-48.037809</td>
      <td>13.889753</td>
      <td>41.383216</td>
      <td>-4.055654</td>
      <td>-41.357420</td>
      <td>-94.853001</td>
      <td>90.0</td>
      <td>10.258214</td>
    </tr>
    <tr>
      <th>6</th>
      <td>65.962191</td>
      <td>-5.110247</td>
      <td>41.383216</td>
      <td>-4.055654</td>
      <td>-41.357420</td>
      <td>-170.853001</td>
      <td>365.0</td>
      <td>6.928216</td>
    </tr>
    <tr>
      <th>7</th>
      <td>65.962191</td>
      <td>-5.110247</td>
      <td>41.383216</td>
      <td>-4.055654</td>
      <td>-41.357420</td>
      <td>-170.853001</td>
      <td>28.0</td>
      <td>-0.321784</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-48.037809</td>
      <td>13.889753</td>
      <td>41.383216</td>
      <td>-4.055654</td>
      <td>-41.357420</td>
      <td>-94.853001</td>
      <td>28.0</td>
      <td>9.078214</td>
    </tr>
    <tr>
      <th>9</th>
      <td>160.962191</td>
      <td>-100.110247</td>
      <td>41.383216</td>
      <td>-4.055654</td>
      <td>-41.357420</td>
      <td>-170.853001</td>
      <td>28.0</td>
      <td>2.518216</td>
    </tr>
  </tbody>
</table>
</div>



### Registering Python Functions

Now that we have a working transform the next step is to register it with Abacus.AI to allow it to run the function when required by workflows. For simple self-contained functions we can just pass the function to the client and it will build a suitable resource to ship the python code to Abacus.AI. For more complicated functions and in cases where additional resources are required you can instead build an archive and add it to the registration function.

Registering the function involves supplying the source artifact, the name of the function implementing the transform and a list of required input feature groups. These feature groups will be passed as dataframe arguments to the functions. Optionally, you can also supply configuration options as keywork arguments that can alter the behavior of the function. For example, the same function may be used to construct two different feature groups differing only in the keyword arguments.

Note, that Abacus.AI will ensure that the function is operating on the latest versions of the input feature groups.


```python
concrete_flyash = client.create_feature_group_from_function(
    table_name='concrete_by_flyash',
    function_source_code=separate_by_flyash,
    function_name='separate_by_flyash',
    input_feature_groups=['concrete_strength'])
```


```python
concrete_flyash.create_version()
concrete_flyash.wait_for_materialization()
concrete_by_flyash_df = concrete_flyash.load_as_pandas()
```

### Custom Model
Now we will define a custom model trained on this flyash partitioned data. A custom training function is similar in many ways to a custom transform. The main difference being instead of returning a new dataframe with the transformed data it returns an object containing the trained model. It is required that object returned should be pickleable by the standard python `pickle` library. However, the model is free to serialize additional data to local disk in the current working directory. The contents of the working directory will be made available at prediction time. There is support for supplying an initialization function along with prediction function that will receive the unpickled model object and transform it based on data loaded from disk to use at prediction. This will be covered in more detail later.

To illustrate that the training can be customized arbitrarily we will train a composite model that depending on the age of the concrete uses a linear model on quantized features or a GBDT trained on raw inputs.


```python
!pip install catboost
```
> Collecting catboost
> 
> Downloading catboost-1.0.0-cp37-none-manylinux1_x86_64.whl (76.4 MB)
> 
> |===========================| 76.4 MB 36 kB/s 
> 
> Requirement already satisfied: pandas>=0.24.0 in /usr/local/lib/python3.7/dist-packages (from catboost) (1.1.5)
> 
> Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from catboost) (1.4.1)
> 
> Requirement already satisfied: graphviz in /usr/local/lib/python3.7/dist-packages (from catboost) (0.10.1)
> 
> Requirement already satisfied: numpy>=1.16.0 in /usr/local/lib/python3.7/dist-packages (from catboost) (1.19.5)
> 
> Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from catboost) (3.2.2)
> 
> Requirement already satisfied: plotly in /usr/local/lib/python3.7/dist-packages (from catboost) (4.4.1)
> 
> Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from catboost) (1.15.0)
> 
> Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.24.0->catboost) (2.8.2)
> 
> Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.24.0->catboost) (2018.9)
> 
> Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->catboost) (2.4.7)
> 
> Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->catboost) (0.10.0)
> 
> Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->catboost) (1.3.2)
> 
> Requirement already satisfied: retrying>=1.3.3 in /usr/local/lib/python3.7/dist-packages (from plotly->catboost) (1.3.3)
> 
> Installing collected packages: catboost
> 
> Successfully installed catboost-1.0.0


Just like with data transforms we can test our function locally to ensure it works on the data frame as expected and that it is building a reasonable model. Notice that the model object we return is tuple comprising
- columns used as inputs to the sub models
- the quantile transform
- linear model
- catboost model

Since this tuple can be pickled we do not need to bother writing anything to local disk. Also we will be able to use the default identity initialization function which will just return this tuple unmodified at prediction time.


```python
def train(concrete_by_flyash):
  # set the seed for reproducible results
  import numpy as np
  np.random.seed(5)

  X = concrete_by_flyash.drop(['csMPa'], axis=1)
  y = concrete_by_flyash.csMPa
  recent = concrete_by_flyash.age < 10
  from sklearn.preprocessing import QuantileTransformer
  from sklearn.linear_model import LinearRegression
  qt = QuantileTransformer(n_quantiles=20)
  recent_model = LinearRegression()
  _ = recent_model.fit(qt.fit_transform(X[recent]), y[recent])
  print(f'Linear model R^2 = {recent_model.score(X[recent], y[recent])}')

  from catboost import Pool, CatBoostRegressor
  train_pool = Pool(X[~recent], y[~recent])
  older_model = CatBoostRegressor(iterations=5, depth=2, loss_function='RMSE')
  _ = older_model.fit(train_pool)
  metrics = older_model.eval_metrics(train_pool, ['RMSE'])
  old_r2 = 1 - metrics['RMSE'][-1]**2 / y[~recent].var()
  print(f'Catboost model R^2 = {old_r2}')

  return (X.columns, qt, recent_model, older_model)

local_model = train(concrete_by_flyash_df)
```

> Linear model R^2 = -59474.80409065778
> 
> Learning rate set to 0.5
> 
> 0:	learn: 12.7627412	total: 46.9ms	remaining: 188ms
> 
> 1:	learn: 11.5585084	total: 47.7ms	remaining: 71.6ms
> 
> 2:	learn: 10.3223491	total: 48.4ms	remaining: 32.3ms
> 
> 3:	learn: 9.3247540	total: 49ms	remaining: 12.3ms
> 
> 4:	learn: 8.5430952	total: 49.6ms	remaining: 0us
> 
> Catboost model R^2 = 0.6814947748102853


### Prediction Function

To actually use this model for predictions we need to tell the service how to evaluate a new input against the returned model object. This function could be as simple as calling `predict()` on a scikit-learn compliant model. However, it will usually be the case that there will be some translation of the request data into model inputs prior to the final invocation. This is to match any feature engineering / transformation done inside the training function. Keep in mind any feature transformation done in feature group transformations will be handled automatically by the service for batch predictions.

In the example we are building there is even more complexity because the model is a composite model built on two partitions of the data so the prediction function needs to dispatch the input to the right model based on one of the input features.

We can follow the same pattern of testing locally to ensure that the prediction function works as expected. If the model requires an initialization function that loads data from disk it would also be good to test that locally.


```python
def predict(model, query):
  # abacusai.get_client().get_feature_group().lookup(...)
  columns, qt, recent_model, older_model = model
  import pandas as pd
  X = pd.DataFrame({c: [query[c]] for c in columns})
  if X.age[0] < 10:
    y = recent_model.predict(qt.transform(X))[0]
  else:
    y = older_model.predict(X.values.reshape(-1))
  return {'csMPa': y}

for _, r in concrete_by_flyash_df[concrete_by_flyash_df.age < 10][:5].iterrows():
  print(predict(local_model, r.to_dict()), r['csMPa'])

for _, r in concrete_by_flyash_df[concrete_by_flyash_df.age > 10][:5].iterrows():
  print(predict(local_model, r.to_dict()), r['csMPa'])
```


> {'csMPa': -31.75412474980192} -28.711784452296826
> 
> {'csMPa': -5.324797742032455} 1.8282155477031736
> 
> {'csMPa': -4.377726654712578} -1.6917844522968295
> 
> {'csMPa': -23.147157848108026} -21.721784452296827
> 
> {'csMPa': -16.712019233341156} -10.511784452296826
> 
> {'csMPa': 19.340295162698652} 43.21821554770317
> 
> {'csMPa': 19.340295162698652} 25.118215547703173
> 
> {'csMPa': 10.376285866501192} 3.4982155477031753
> 
> {'csMPa': 10.376285866501192} 4.278215547703169
> 
> {'csMPa': -2.273645085750303} 7.528215547703169

### Register the Model
We now put together the feature group, the training function and the prediction function as a new Abacus model. Like with custom feature groups the model has to specify the feature groups required for training which will be passed as arguments to the train function.


```python
model = client.create_model_from_functions(project_id=project, 
                                   train_function=train, 
                                   predict_function=predict, 
                                   training_input_tables=['concrete_by_flyash'])
```

Wait for the model to finish training and then deploy the model to use for prediction.


```python
model.wait_for_training()
deployment_token = client.create_deployment_token(project.project_id).deployment_token
deployment = client.create_deployment(model_id=model.model_id)
deployment.wait_for_deployment()
```


> Linear model R^2 = -59474.80409065778
> 
> Learning rate set to 0.5
> 
> 0:	learn: 12.7627412	total: 573us	remaining: 2.29ms
> 
> 1:	learn: 11.5585084	total: 1.03ms	remaining: 1.54ms
> 
> 2:	learn: 10.3223491	total: 1.73ms	remaining: 1.15ms
> 
> 3:	learn: 9.3247540	total: 2.41ms	remaining: 601us
> 
> 4:	learn: 8.5430952	total: 3ms	remaining: 0us
> 
> Catboost model R^2 = 0.6814947748102853


Now we can run predictions on Abacus and compare against predictions from the local model.


```python
# locally trained
for _, r in concrete_by_flyash_df[concrete_by_flyash_df.age < 10][:5].iterrows():
  print(predict(local_model, r.to_dict()), r['csMPa'])

print(' Is equal to ')

# remotely trained
for _, r in concrete_by_flyash_df[concrete_by_flyash_df.age < 10][:5].iterrows():
  print(client.predict(deployment_token, deployment.deployment_id, r.to_dict()), r['csMPa'])
```


> {'csMPa': -31.75412474980192} -28.711784452296826
> 
> {'csMPa': -5.324797742032455} 1.8282155477031736
> 
> {'csMPa': -4.377726654712578} -1.6917844522968295
> 
> {'csMPa': -23.147157848108026} -21.721784452296827
> 
> {'csMPa': -16.712019233341156} -10.511784452296826
> 
>   Is equal to
> 
> {'csMPa': -31.75412474980192} -28.711784452296826
> 
> {'csMPa': -5.324797742032455} 1.8282155477031736
> 
> {'csMPa': -4.377726654712578} -1.6917844522968295
> 
> {'csMPa': -23.147157848108026} -21.721784452296827
> 
> {'csMPa': -16.712019233341156} -10.511784452296826


### Setup Batch Predictions

We can setup a new dataset to feed a batch prediction job. Abacus will run the prediction dataset through the feature transformation function and then apply the custom model to generate predictions for the uploaded data. Keep in mind the input to the model will be what is generated by transform. The inputs to the model are included in the batch prediction download along with model outputs.


```python
try: 
  prediction_dataset = client.describe_dataset(client.describe_feature_group_by_table_name('concrete_strength_prediction_input').dataset_id)
except ApiException: # dataset not found
  prediction_dataset = client.create_dataset_from_file_connector(
      name='Concrete Strength Prediction Input',
      table_name='concrete_strength_prediction',
      location='s3://abacusai.exampledatasets/predicting/concrete_measurements.csv')
  prediction_dataset.wait_for_inspection()

batch_prediction = client.create_batch_prediction(deployment.deployment_id)
batch_prediction.set_batch_prediction_dataset_remap({
    concrete_dataset.dataset_id: prediction_dataset.dataset_id
})
batch_prediction_run = batch_prediction.start()
batch_prediction_run.wait_for_predictions()
with open('batch_predictions_results.csv', 'wb') as bpr_file:
  batch_prediction_run.download_result_to_file(bpr_file)
!head batch_predictions_results.csv
```

### Attach Refresh Schedules

As a final step we can attach refresh schedules to various objects to ensure that they are updated regularly without any manual intervention. This allows the custom model to run with the same level of automation as models generated internally by the service.


```python
concrete_dataset.set_refresh_schedule('0 4 * * 1')
model.set_refresh_schedule('0 6 * * 1')
deployment.set_auto_deployment(True)
batch_prediction.set_refresh_schedule('0 8 * * 1')
```
