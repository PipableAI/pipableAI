from dataclasses import dataclass

import psycopg2
from pandas import DataFrame

from ..interfaces.database_connector_interface import DatabaseConnectorInterface


@dataclass
class PostgresConfig:
    """Data class for PostgreSQL connection configuration.

    Args:
        host (str): The hostname or IP address of the PostgreSQL server.
        port (int): The port number to connect to on the PostgreSQL server.
        database (str): The name of the PostgreSQL database.
        user (str): The username for connecting to the PostgreSQL server.
        password (str): The password for the specified username.

    Attributes:
        host (str): The hostname or IP address of the PostgreSQL server.
        port (int): The port number to connect to on the PostgreSQL server.
        database (str): The name of the PostgreSQL database.
        user (str): The username for connecting to the PostgreSQL server.
        password (str): The password for the specified username.
    """

    host: str
    port: int
    database: str
    user: str
    password: str


class PostgresConnector(DatabaseConnectorInterface):
    """A class for establishing and managing the PostgreSQL database connection.

    This class provides methods for connecting to a remote PostgreSQL server and executing SQL queries.
    It uses the `psycopg2` library for database interaction.

    Args:
        config (PostgresConfig): The configuration for connecting to the PostgreSQL server.

    Attributes:
        config (PostgresConfig): The configuration for connecting to the PostgreSQL server.
        connection (psycopg2.extensions.connection): The connection to the PostgreSQL server.
        cursor (psycopg2.extensions.cursor): The cursor for executing SQL queries.

    Raises:
        ConnectionError: If failed to connect to the PostgreSQL server.
        ValueError: If an error occurs during query execution.

    Note:
        The `execute_query` method returns the query results as a Pandas DataFrame.

    Warning:
        Ensure to disconnect from the database using the `disconnect` method after executing queries
        to release resources.

    Example:
        To establish a connection and execute a query, create an instance of `PostgresConnector` and
        call the `execute_query` method.

        .. code-block:: python

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
        """Establish a connection to the PostgreSQL server."""
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
            raise ConnectionError(
                f"Failed to connect to the PostgreSQL server: {str(e)}"
            )

    def disconnect(self):
        """Close the connection to the PostgreSQL server."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def execute_query(self, query: str) -> DataFrame:
        """Execute an SQL query on the connected PostgreSQL server and return the result as
        a Pandas DataFrame.

        Args:
            query (str): The SQL query to execute.

        Returns:
            DataFrame: A Pandas DataFrame representing the query results.

        Raises:
            ValueError: If an error occurs during query execution.
        """
        try:
            self.cursor.execute(query)
            columns = [desc[0] for desc in self.cursor.description]
            data = self.cursor.fetchall()
            df = DataFrame(data, columns=columns)
            return df
        except psycopg2.Error as e:
            raise ValueError(f"SQL query execution error: {e}")


__all__ = ["PostgresConfig", "PostgresConnector"]
