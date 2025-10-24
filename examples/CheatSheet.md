## Abacus Python SDK Usage
Abacus.AI provides a comprehensive Python SDK that can be combined with Open Source python libraries to complement your solutions. Within our Python SDK we also have specific methods that will allow you to:
1. Create AI Workflows within the platform
2. Create Models (GenAI or ML)

### Basic Commands
Here is a list of some of the basic methods to use:

```python
# Connect to Abacus.AI
from abacusai import ApiClient
client = ApiClient('YOUR_API_KEY')
```

```python
# Below method returns methods related to the query of the user
client.suggest_abacus_apis(query = 'API Methods to invoke a LLM', verbosity=1, limit=5)
```

```python
# Below method allows you to Invoke a LLM. json_response_schema is optional - in cases where a JSON response is required.
r = client.evaluate_prompt(prompt = "In this course, you will learn about car batteries, car doors, and car suspension system",
                           # system_message = "OPTIONAL, but good to have", 
                           llm_name = 'OPENAI_GPT4O',
                           response_type='json',
                           json_response_schema = {"learning_objectives": {"type": "list", "description": "A list of learning objectives", "is_required": True}}
)
```

```python
# Below method allows you to do OCR on any type of input (PDF, WORD, MD, etc)
from abacusai.client import BlobInput
document = BlobInput.from_local_file("YOUR_DOCUMENT.pdf/word/etc")
extracted_doc_data = client.extract_document_data(document.contents)

# print first 100 chracters of page 0
print(extracted_doc_data.pages[0][0:100])
# print first 100 chracters of all embedded text
print(extracted_doc_data.embedded_text[0:100])
```

```python
# Below Method allows you to load an existing Feature group from the Abacus platform
df = client.describe_feature_group_by_table_name('YOUR_FEATURE_GROUP_NAME').load_as_pandas()
```

```python
# Below method allows you to query data directly from a DB connector that has been established within Abacus.

```

### Best Practices and when to use the Python SDK
Abacus combines infrastructure with common GenAI methods required to build applications. You're still free to use open-source libraries within your code, combining our powerful built-in methods with existing tools and frameworks.

| Method Signature | Explanation |
|------------------|-------------|
| `suggest_abacus_apis(query: str, verbosity: int, limit: int)` | Describe what you need, and we will return the methods that will help you achieve it. |
| `describe_project(project_id: str)` | Describes a project |
| `create_dataset_from_upload(name: str, table_name: str, upload_id: str)` | Creates a dataset object from local data |
| `describe_feature_group_by_table_name(table_name: str)` | Describes the feature group using the table name |
| `list_models(project_id: str, limit: int)` | Lists models of a project |
| `get_conversation_response(deployment_conversation_id: str, message: str)` | Uses a chatLLM deployment with conversation history. Useful when you need to use the API. You create a conversation ID and send it or use the one created by Abacus. |
| `evaluate_prompt(prompt: str, system_message: str, llm_name: str)` | LLM call for a user query. Can get JSON output using additional arguments |
| `get_matching_documents(deployment_id: str, query: str)` | Gets the search results for a user query using document retriever directly. Can be used along with evaluate_prompt to create a customized chat LLM-like agent |
| `extract_document_data(doc_id: str, document_processing_config)` | Extracts data from a PDF, Word document, etc. using OCR or the digital text. |
| `query_database_connector(database_connector_id: str, query: str)` | Executes a SQL query on top of a database connector. Will only work for connectors that support it. |
