# pipable/pipable.py
"""Pipable: A Python package for connecting to a remote PostgreSQL server, generating and executing the natural langauge based data 
search queries which are mapped to SQL queries using a using the pipLLM.

This module provides classes and functions for connecting to a PostgreSQL database and using a language model to generate SQL queries.

Author: Your Name
"""

from core.postgresql_connector import PostgresConfig, PostgresConnector
from llm_client.pipllm import PipLlmApiClient

import pandas as pd

class Pipable:
    """A class for connecting to a remote PostgreSQL server and generating SQL queries.

    This class provides methods for establishing a connection to a remote PostgreSQL server and using a language model to generate SQL queries.
    """
    def __init__(self, postgres_config: PostgresConfig, llm_api_base_url: str):
        """Initialize a Pipable instance.

        Args:
            postgres_config (PostgresConfig): The configuration for connecting to the PostgreSQL server.
            llm_api_base_url (str): The base URL of the language model API.
        """
        self.postgres_config = postgres_config
        self.llm_api_client = PipLlmApiClient(llm_api_base_url)
        self.connected = False
        self.connection = None
        self.all_table_queries = None  # Store create table queries for all tables

    def _generate_sql_query(self, context, question):
        generated_text = self.llm_api_client.generate_text(context, question)
        if not generated_text:
            raise ValueError("LLM did not generate a valid SQL query.")
        return generated_text.strip()

    def connect(self):
        """Establish a connection to the PostgreSQL server.

        This method establishes a connection to the remote PostgreSQL server using the provided configuration.

        Raises:
            ConnectionError: If the connection to the server cannot be established.
        """
        if not self.connected:
            self.connection = PostgresConnector(self.postgres_config)
            self.connection.connect()
            self.connected = True

    def disconnect(self):
        """Close the connection to the PostgreSQL server.

        This method closes the connection to the remote PostgreSQL server.
        """
        if self.connected:
            self.connection.disconnect()
            self.connected = False

    def retrieve_all_tables(self):
        # Query to retrieve all table names in the database
        query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_type = 'BASE TABLE';
        """

        result = self.connection.execute_query(query)
        table_names = result["table_name"].tolist()

        # Generate create table queries for all tables
        self.all_table_queries = {}
        for table_name in table_names:
            query = f"SHOW CREATE TABLE {table_name};"
            result = self.connection.execute_query(query)
            create_table_query = result.iloc[0]["create_statement"]
            self.all_table_queries[table_name] = create_table_query

    def ask(self, context=None, question=None):
        """Generate an SQL query and execute it on the PostgreSQL server.

        Args:
            context (str, optional): The context or CREATE TABLE statements for the query. If not provided, it will be auto-generated.
            question (str, optional): The query to perform in simple English.

        Returns:
            pandas.DataFrame: A DataFrame containing the query result.

        Raises:
            ValueError: If the language model does not generate a valid SQL query.
        """
        try:
            # Connect to PostgreSQL if not already connected
            self.connect()

            if context is None:
                # Retrieve create table queries for all tables if context is not provided
                if self.all_table_queries is None:
                    self.retrieve_all_tables()

                # Concatenate create table queries for all tables into context
                context = "\n".join(self.all_table_queries.values())

            # Generate SQL query from LLM
            sql_query = self._generate_sql_query(context, question)

            # Execute SQL query
            result_df = self.connection.execute_query(sql_query)

            return result_df
        except Exception as e:
            raise Exception(f"Error in 'ask' method: {str(e)}")
