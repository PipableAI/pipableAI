from abc import ABC, abstractmethod

from pandas import DataFrame


class DatabaseConnectorInterface(ABC):
    """Abstract base class for database connector interfaces.

    This class defines the interface for database connectors. Concrete implementations
    must inherit from this class and provide implementations for the abstract methods.

    Attributes:
        None

    Methods:
        - connect(): Establish a connection to the database.
        - disconnect(): Close the connection to the database.
        - execute_query(query: str) -> DataFrame: Execute an SQL query and return the result as a Pandas DataFrame.

    Example:
        To create a custom database connector, inherit from this class and provide implementations
        for the abstract methods.

        .. code-block:: python

            from abc import ABC, abstractmethod
            from pandas import DataFrame

            class CustomDatabaseConnector(DatabaseConnectorInterface):
                def __init__(self, config):
                    # Initialize the connector with configuration
                    pass

                def connect(self):
                    # Implement connection logic
                    pass

                def disconnect(self):
                    # Implement disconnection logic
                    pass

                def execute_query(self, query: str) -> DataFrame:
                    # Implement query execution logic and return the result as a DataFrame
                    pass
    """

    @abstractmethod
    def connect(self):
        """Establish a connection to the database."""
        pass

    @abstractmethod
    def disconnect(self):
        """Close the connection to the database."""
        pass

    @abstractmethod
    def execute_query(self, query: str) -> DataFrame:
        """Execute an SQL query on the connected database and return the result as
        a Pandas DataFrame.

        Args:
            query (str): The SQL query to execute.

        Returns:
            DataFrame: A Pandas DataFrame representing the query results.
        """
        pass


__all__ = ["DatabaseConnectorInterface"]
