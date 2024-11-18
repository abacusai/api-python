# Abacus.AI Python API Client [![PyPI version](https://badge.fury.io/py/abacusai.svg)](https://badge.fury.io/py/abacusai)

The official Python API Client Library for Abacus.AI.

## Installation

Install using pip:

```console
$ pip install abacusai
```

## Quick Start

### Authentication

To use the API, you'll need an API key from your Abacus.AI account. Initialize the client:

```python
from abacusai import ApiClient
client = ApiClient('YOUR_API_KEY')
```

### Basic Usage Examples

```python
# List your projects
projects = client.list_projects()

# Create a new project
project = client.create_project("My Project")

# Get project details
project_info = client.get_project(project['projectId'])
```

## Documentation

- API Reference: https://abacusai.github.io/api-python/autoapi/abacusai/index.html
- Full Documentation: https://abacus.ai/app/help/ref/overview
- Examples & Tutorials: https://abacus.ai/app/help/examples

## Features

- Complete API coverage for Abacus.AI platform
- Type hints for better IDE support
- Automatic retries and error handling
- Async support
- Comprehensive logging

## Requirements

- Python 3.7+
- requests>=2.25.0
- pandas>=1.0.0
- numpy>=1.19.0

## Error Handling

```python
from abacusai.errors import AbacusAIError

try:
    client.get_project("non_existent_id")
except AbacusAIError as e:
    print(f"Error: {e.message}")
    print(f"Status code: {e.status_code}")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m pytest`
5. Submit a pull request

## Support

- Email: support@abacus.ai
- Documentation: https://abacus.ai/app/help
- GitHub Issues: https://github.com/abacusai/api-python/issues

## License

[MIT](https://github.com/abacusai/api-python/blob/main/LICENSE)
