from typing import List, Optional

from pandas import DataFrame

from .core.dev_logger import dev_logger
from .core.postgresql_connector import PostgresConfig, PostgresConnector
from .interfaces.database_connector_interface import DatabaseConnectorInterface
from .interfaces.llm_api_client_interface import LlmApiClientInterface
from .llm_client.pipllm import PipLlmApiClient


class Pipable:
    """A Python package for connecting to a remote PostgreSQL server, generating and executing natural language-based data search queries mapped to SQL queries using the pipLLM.

    This module provides classes and functions for connecting to a PostgreSQL database and using a language model to generate SQL queries.

    Attributes:
        database_connector (DatabaseConnectorInterface): The database connector implementing the DatabaseConnectorInterface.
        llm_api_client (LlmApiClientInterface): The API client implementing the LlmApiClientInterface.
        connected (bool): A boolean indicating if the Pipable instance is connected to the database.
        connection: The connection object to the remote PostgreSQL server.
        logger: The logger object for logging messages and errors.
        all_table_queries (list): A list to store CREATE TABLE queries for all tables in the database.
    """

    def __init__(
        self,
        database_connector: DatabaseConnectorInterface,
        llm_api_client: LlmApiClientInterface,
    ):
        """Initialize a Pipable instance.

        Args:
            database_connector (DatabaseConnectorInterface): The configuration for connecting to the PostgreSQL server.
            llm_api_client (LlmApiClientInterface): The API client for generating SQL queries using the language model.
        """
        self.database_connector = database_connector
        self.llm_api_client = llm_api_client
        self.connected = False
        self.connection = None
        self.logger = dev_logger()
        self.logger.info("logger initialized in Pipable")
        self.all_table_queries = None  # Store create table queries for all tables

    def _generate_sql_query(self, context, question):
        self.logger.info("generating query using llm")
        generated_text = self.llm_api_client.generate_text(context, question)
        if not generated_text:
            self.logger.error(f"LLM failed to generate a SQL query: {e}")
            raise ValueError("LLM failed to generate a SQL query.")
        return generated_text.strip()

    def connect(self):
        """Establish a connection to the Database server.

        This method establishes a connection to the remote PostgreSQL server using the provided database connector.

        Raises:
            ConnectionError: If the connection to the server cannot be established.
        """
        if not self.connected:
            try:
                self.database_connector.connect()
                self.connected = True
            except Exception as e:
                self.logger.error(f"Failed to connect to the database: {str(e)}")
                raise ConnectionError("Failed to connect to the database.")

    def disconnect(self):
        """Close the connection to the Database server.

        This method closes the connection to the remote PostgreSQL server.
        """
        if self.connected:
            try:
                self.database_connector.disconnect()
                self.connected = False
            except Exception as e:
                self.logger.error(f"Failed to disconnect from the database: {str(e)}")
                raise ConnectionError("Failed to disconnect from the database.")

    def _generate_create_table_statements(
        self, table_names: Optional[List[str]] = None
    ):
        """
        Generate CREATE TABLE statements for the specified tables or all tables.

        Parameters:
            table_names (list, optional): The list of table names for the query context.
                If not provided, it will be auto-generated.

        Returns:
            list: A list of CREATE TABLE statements.
        """
        if self.all_table_queries is not None:
            return self.all_table_queries
        # Check if specific table names are provided, else get all tables
        if table_names is not None and len(table_names) > 0:
            tables_to_fetch = ",".join([f"'{table}'" for table in table_names])
            where_clause = f"WHERE table_name IN ({tables_to_fetch})"
        else:
            where_clause = "WHERE table_schema = 'public'"

        # SQL query to extract column names and data types
        column_info_query = f"""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        {where_clause};
        """

        try:
            # Execute the SQL query using the database connector and get the result as DataFrame
            column_info_df = self.database_connector.execute_query(column_info_query)

            # Group column info by table name using Pandas groupby
            grouped_columns = column_info_df.groupby("table_name").apply(
                lambda x: ", ".join(
                    [
                        f"{row['column_name']} {row['data_type']}"
                        for _, row in x.iterrows()
                    ]
                )
            )

            # Generate CREATE TABLE statements in Python
            self.all_table_queries = [
                f"CREATE TABLE {table_name} ({columns});"
                for table_name, columns in grouped_columns.items()
            ]

            return self.all_table_queries
        except Exception as e:
            self.logger.error(f"Error generating CREATE TABLE statements: {str(e)}")
            raise ValueError("Error generating CREATE TABLE statements.")

    def ask_and_execute(
        self, question: str, table_names: Optional[List[str]]
    ) -> DataFrame:
        """Generate an SQL query and execute it on the PostgreSQL server.

        Args:
            table_names (list, optional): The list of table names for the query context.
            If not provided, it will be auto-generated.
            question (str): The query to perform in simple English.

        Returns:
            pandas.DataFrame: A DataFrame containing the query result.

        Raises:
            ValueError: If the language model does not generate a valid SQL query.
        """
        try:
            # Connect to PostgreSQL if not already connected
            self.connect()

            # Generate CREATE TABLE statements for the specified tables or all tables
            create_table_statements = self._generate_create_table_statements(
                table_names
            )
            # Concatenate create table statements into a single line for context
            context = " ".join(create_table_statements)

            # Generate SQL query from LLM
            sql_query = self._generate_sql_query(context, question)

            # Execute SQL query
            result_df = self.database_connector.execute_query(sql_query)

            return result_df
        except Exception as e:
            raise ValueError(f"Error in 'ask_and_execute' method: {str(e)}")

    def ask(self, question: str, table_names: Optional[List[str]] = None) -> str:
        """Generate an SQL query.

        Args:
            table_names (list, optional): The list of table names for the query context.
            If not provided, it will be auto-generated.
            question (str): The query to perform in simple English.

        Returns:
            str: A sql query result.

        Raises:
            ValueError: If the language model does not generate a valid SQL query.
        """
        try:
            # Connect to PostgreSQL if not already connected
            self.connect()

            # Generate CREATE TABLE statements for the specified tables or all tables
            create_table_statements = self._generate_create_table_statements(
                table_names
            )
            # Concatenate create table statements into a single line for context
            context = " ".join(create_table_statements)

            # Generate SQL query from LLM
            sql_query = self._generate_sql_query(context, question)

            return sql_query
        except Exception as e:
            raise ValueError(f"Error in 'ask' method: {str(e)}")
