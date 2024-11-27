## Python SDK Documentation Examples
The full Documentation of the Abacus.AI python SDK can be found [here](https://abacusai.github.io/api-python/autoapi/abacusai/index.html)

Please note that from within the platform's UI, you will have access to template/example code for all below cases:
- [Python Feature Group](https://abacus.ai/app/python_functions_list)
- [Pipelines](https://abacus.ai/app/pipelines)
- [Custom Loss Functions](https://abacus.ai/app/custom_loss_functions_list)
- Custom Models & [Algorithms](https://abacus.ai/app/algorithm_list)
- [Python Modules](https://abacus.ai/app/modules_list)
- And others...

Furthermore, the platform's `AI Engineer`, also has access to our API's and will be able to provide you with any code that you require.

#### Abacus.AI - API Command Cheatsheet
Here is a list of important methods you should keep saved somewhere. You can find examples for all of these methods inside the notebooks, or you can refer to the official documentation.

```python
from abacusai import ApiClient
client = ApiClient('YOUR_API_KEY')
#client.whatever_name_of_method_below()
```

| Method                                             | Explanation                                                                                                                                                                  |
|----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| suggest_abacus_apis                                | Describe what you need, and we will return the methods that will help you achieve it.                                                                                        |
| describe_project                                   | Describe's project                                                                                                                                                           |
| create_dataset_from_upload                         | Creates a dataset object from local data                                                                                                                                     |
| describe_feature_group_by_table_name               | Describes the feature group using the table name                                                                                                                             |
| describe_feature_group_version                     | Describes the feature group using the feature group version                                                                                                                  |
| list_models                                        | List's models of a project                                                                                                                                                   |
| extract_data_using_llm                             | Extracts data from a document. Allows you to create a json output and extract specific information from a document                                                           |
| execute_data_query_using_llm                       | Runs SQL on top of feature groups based on natural language input. Can return both SQL and the result of SQL execution.                                                      |
| get_chat_response                                  | Uses a chatLLM deployment. Can be used to add filters, change LLM and do advanced use cases using an agent on top of a ChatLLM deployment.                                   |
| get_chat_response_with_binary_data                 | Same as above, but you can also send a binary dataset                                                                                                                        |
| get_conversation_response                          | Uses a chatLLM deployment with conversation history. Useful when you need to use the API. You create a conversation ID and you send it or you use the one created by Abacus. |
| get_conversation_response_with_binary_data         | Same as above, but you can also send a binary dataset                                                                                                                        |
| evaluate_prompt                                    | LLM call for a user query. Can get JSON output using additional arguments                                                                                                    |
| get_matching_documents                             | Gets the search results for a user query using document retriever directly. Can be used along with evaluate_prompt to create a customized chat llm like agent                |
| get_relevant_snippets                              | Creates a doc retriever on the fly for retrieving search results                                                                                                             |
| extract_document_data                              | Extract data from a PDF, Word document, etc using OCR or using the digital text.                                                                                             |
| get_docstore_document                              | Download document from the doc store using their doc_id.                                                                                                                     |
| get_docstore_document_data                         | Get extracted or embedded text from a document using their doc_id.                                                                                                                         |
| stream_message                                     | Streams message on the UI for agents                                                                                                                                         |
| update_feature_group_sql_definition                | Updates the SQL definition of a feature group                                                                                                                                |
| query_database_connector                           | Executes a SQL query on top of a database connector. Will only work for connectors that support it.                                                                          |
| export_feature_group_version_to_file_connector     | Exports a feature group to a file connector                                                                                                                                  |
| export_feature_group_version_to_database_connector | Exports a feature group to a database connector                                                                                                                              |
| create_dataset_version_from_file_connector         | Refreshes data from the file connector connected to the file connector.                                                                                                      |
| create_dataset_version_from_database_connector     | Refreshes data from the file connector connected to the database connector.                                                                                                  |
