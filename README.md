# Pipable 🚀

## About

Pipable is a Python package that simplifies the process of querying a PostgreSQL database. It provides a high-level interface for connecting to a remote PostgreSQL server, generating and executing natural language-based data search queries mapped to SQL queries using a language model.

## Features

- **Natural Language Queries**: Express database queries in plain English.
- **PostgreSQL Integration**: Seamlessly connects to PostgreSQL databases.
- **Language Model Mapping**: Translates natural language queries into SQL queries.
- **Structured Results**: Returns query results in a structured format for easy processing.

## Installation

To install Pipable, you can use `pip3`, Python's package manager. Open your terminal or command prompt and run the following command:

```bash
pip3 install pipable
```

**Note:** Pipable requires Python 3.7 or higher.

If you prefer to install Pipable from source, you can clone the GitHub repository and install it using `setup.py`. Navigate to the project directory and run:

```bash
python3 setup.py install
```

This will install Pipable locally on your system.

## Usage

Pipable simplifies the process of connecting to a remote PostgreSQL server, generating SQL queries using a language model, and executing them. This section provides a step-by-step guide on how to use Pipable effectively in your Python projects.

### Import Pipable:

To start using Pipable, import the necessary classes and interfaces:

```python
from pipable import Pipable
from pipable.llm_client.pipllm import PipLlmApiClient
from pipable.core.postgresql_connector import PostgresConfig, PostgresConnector
```

### Initialize Pipable:

Create an instance of Pipable by providing the required database configuration and LLM API base URL:

```python
# Define PostgreSQL configuration
postgres_config = PostgresConfig(
    host="your_postgres_host",
    port=5432,  # Replace with your port number
    database="your_database_name",
    user="your_username",
    password="your_password",
)

# Initialize the database connector and LLM API client
database_connector = PostgresConnector(postgres_config)
llm_api_client = PipLlmApiClient(api_base_url="https://your-pipllm-api-url.com")

# Create a Pipable instance
pipable_instance = Pipable(database_connector=database_connector, llm_api_client=llm_api_client)
```

### Generate and Execute Queries:

Generate SQL queries using the language model and execute them on the database.
For Better Performance one can pass the tables names which shold be used in the query

#### When `table_names` is an empty list:

```python
# Generate a query using the language model
table_names = []
question = "List all employees."
try:
    # Generate and execute the query
    result_df = pipable_instance.ask_and_execute(question, table_names)
    print("Query Result:")
    print(result_df)
except Exception as e:
    print(f"Error: {e}")
```

#### When `table_names` is None or not passed in:

```python
# Generate a query using the language model
table_names = None
question = "List all employees."
try:
    # Generate and execute the query
    result_df = pipable_instance.ask_and_execute(question)
    print("Query Result:")
    print(result_df)
except Exception as e:
    print(f"Error: {e}")
```

#### When `table_names` is populated with correct table names:

```python
# Generate and execute a query using the language model
table_names = ["table1", "table2", "table3"]
question = "List all employees."
try:
    # Generate and execute the query
    result_df = pipable_instance.ask_and_execute(question, table_names)
    print("Query Result:")
    print(result_df)
except Exception as e:
    print(f"Error: {e}")

# Generate a query using the language model
table_names = ["table1", "table2", "table3"]
question = "List all employees."
try:
    # Generate and execute the query
    result_query = pipable_instance.ask(question, table_names)
    print("Query Result:")
    print(result_query)
except Exception as e:
    print(f"Error: {e}")
```

Handle exceptions appropriately to ensure graceful error handling in your application.

### Disconnect from the Database:

Close the connection to the PostgreSQL server after executing the queries:

```python
pipable_instance.disconnect()
```

or

```python
database_connector.disconnect()
```

Ensure that you disconnect from the database to release resources when the queries are completed.

### Additional Information:

- Check the interfaces: `DatabaseConnectorInterface` and `LlmApiClientInterface` for more details on the methods and functionalities provided by Pipable.

## Example

For a complete example, refer to the `example/` directory in this repository.

## Documentation

For detailed usage instructions and examples, please refer to the [official documentation](https://pipableai.github.io/pipable-docs/).

## Contributing

We welcome contributions from the community! To contribute to Pipable, follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature-name`
3. Make changes and commit: `git commit -m 'Description of changes'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

Please read our [Contribution Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the Apache 2.0 - see the [LICENSE](LICENSE) file for details.

---