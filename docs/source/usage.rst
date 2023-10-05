Usage
=====

Pipable simplifies the process of connecting to a remote PostgreSQL server, generating SQL queries using a language model, and executing them. This section provides a step-by-step guide on how to use Pipable effectively in your Python projects.

1. **Import Pipable:**

    To start using Pipable, import the necessary classes and interfaces:

    .. code-block:: python

        from pipable import Pipable
        from pipable.llm_client.pipllm import PipLlmApiClient
        from pipable.core.postgresql_connector import PostgresConfig, PostgresConnector

2. **Initialize Pipable:**

    Create an instance of Pipable by providing the required database configuration and LLM API base URL:

    .. code-block:: python

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
        llm_api_client = PipLlmApiClient(api_base_url="https://your-llm-api-url.com")

        # Create a Pipable instance
        pipable_instance = Pipable(database_connector=database_connector, llm_api_client=llm_api_client)

3. **Generate and Execute Queries:**

    Generate SQL queries using the language model and execute them on the database:

    - When `table_names` is an empty list:

    .. code-block:: python

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

    - When `table_names` is None or not passed in:

    .. code-block:: python

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

    - When `table_names` is populated with correct table names:

    .. code-block:: python

        # Generate and Execute a query using the language model
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

   Handle exceptions appropriately to ensure graceful error handling in your application.

4. **Disconnect from the Database:**

    Close the connection to the PostgreSQL server after executing the queries:

    .. code-block:: python

        pipable_instance.disconnect()

    or

    .. code-block:: python

        database_connector.disconnect()

    Ensure that you disconnect from the database to release resources when the queries are completed.

5. **Additional Information:**

    - Check the interfaces: `DatabaseConnectorInterface` and `LlmApiClientInterface` for more details on the methods and functionalities provided by Pipable.

This guide outlines the fundamental steps for using Pipable to interact with a PostgreSQL database. Be sure to customize the error handling and query generation logic based on your specific use case to create a robust and reliable application.

Feel free to explore additional features and methods provided by Pipable to further enhance your database interaction and query generation capabilities.
