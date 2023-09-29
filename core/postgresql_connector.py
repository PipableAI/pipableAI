# pipable/postgresql_connector.py
"""Pipable Connector Module

This module provides classes and functions for connecting to a remote PostgreSQL server.

Classes:
    - PostgresConfig: A data class for PostgreSQL connection configuration.

Functions:
    - connect_to_postgres: Establish a connection to the PostgreSQL server using a provided configuration.
    - disconnect_from_postgres: Close the connection to the PostgreSQL server.

Author: Pipable
"""

import psycopg2
import pandas as pd
from dataclasses import dataclass

@dataclass
class PostgresConfig:
    host: str
    port: int
    database: str
    user: str
    password: str

class PostgresConnector:
    """A class for establishing and managing the PostgreSQL database connection.

    This class provides methods for connecting to a remote PostgreSQL server and executing SQL queries.
    It uses the `psycopg2` library for database interaction.

    Args:
        config (PostgresConfig): The configuration for connecting to the PostgreSQL server.
    
    Example:
        To establish a connection and execute a query, create an instance of `PostgresConnector` and
        call the `execute_query` method.
        
        Example:
        ```python
        from pipable.connector import PostgresConnector, PostgresConfig

        # Define PostgreSQL configuration
        postgres_config = PostgresConfig(
            host="your_postgres_host",
            port=5432,  # Replace with your port number
            database="your_database_name",
            user="your_username",
            password="your_password",
        )

        # Initialize the PostgresConnector instance
        connector = PostgresConnector(postgres_config)

        # Execute a SQL query
        result = connector.execute_query("SELECT * FROM Employees")
        ```
    """
    def __init__(self, config: PostgresConfig):
        """Initialize a PostgresConnector instance.

        Args:
            config (PostgresConfig): The configuration for connecting to the PostgreSQL server.
        """
        self.config = config
        self.connection = None
        self.cursor = None

    def connect(self):
        try:
            self.connection = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
            )
            self.cursor = self.connection.cursor()
        except psycopg2.Error as e:
            raise ConnectionError(f"Failed to connect to the PostgreSQL server: {str(e)}")

    def disconnect(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute an SQL query on the connected PostgreSQL server and return the result as 
        a Pandas DataFrame.

        Args:
            sql_query (str): The SQL query to execute.

        Returns:
            DataFrame: A Pandas DataFrame representing the query results.

        Raises:
            psycopg2.Error: If an error occurs during query execution.
        """
        try:
            self.cursor.execute(query)
            columns = [desc[0] for desc in self.cursor.description]
            data = self.cursor.fetchall()
            df = pd.DataFrame(data, columns=columns)
            return df
        except psycopg2.Error as e:
            raise ValueError(f"SQL query execution error: {str(e)}")

__all__ = ["PostgresConfig", "PostgresConnector"]
