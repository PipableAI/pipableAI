from pipable import Pipable
from pipable.core.postgresql_connector import PostgresConfig, PostgresConnector
from pipable.llm_client.pipllm import PipLlmApiClient

# Define PostgreSQL configuration
postgres_config = PostgresConfig(
    host="localhost",
    port=5432,  # Replace with your port number
    database="sampleDB",
    user="postgres",
    password="postgres",
)

# Initialize the database connector and LLM API client
database_connector = PostgresConnector(postgres_config)
llm_api_client = PipLlmApiClient(api_base_url="http://127.0.0.1:8000")

# Create a Pipable instance
pipable_instance = Pipable(
    database_connector=database_connector,
    llm_api_client=llm_api_client,
)

# Example usage of the ask_and_execute method
table_names = ["actors"]  # Replace with your table names
question = "List first name of all actors."  # Replace with your query question

try:
    # Generate the query
    result_query = pipable_instance.ask(question, table_names)
    print("Query Result:")
    print(result_query)
except Exception as e:
    print(f"Error: {e}")

try:
    # Generate the query without passing the table names
    result_query = pipable_instance.ask(question)
    print("Query Result:")
    print(result_query)
except Exception as e:
    print(f"Error: {e}")

# Disconnect from the database after executing the queries
pipable_instance.disconnect()
