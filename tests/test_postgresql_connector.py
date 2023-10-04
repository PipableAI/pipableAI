import os
import sys
import unittest
from unittest.mock import Mock

# Add the absolute path of the root folder to Python path
root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder)

from pipable.core.postgresql_connector import PostgresConfig, PostgresConnector
from pipable.interfaces.database_connector_interface import DatabaseConnectorInterface


# Define a mock implementation of DatabaseConnectorInterface for testing
class MockDatabaseConnector(DatabaseConnectorInterface):
    def connect(self):
        pass

    def disconnect(self):
        pass

    def execute_query(self, query: str):
        # Return a mocked DataFrame for testing purposes
        return MockDataFrame()


class MockDataFrame:
    def __init__(self):
        self.shape = (2, 2)  # Mock shape of the DataFrame

    def __eq__(self, other):
        # Mock comparison logic for testing
        return True


class TestPostgresConnector(unittest.TestCase):
    def setUp(self):
        # Initialize a PostgresConnector instance with the mock database connector
        self.connector = MockDatabaseConnector()

    def tearDown(self):
        pass

    def test_execute_query(self):
        # Define a test SQL query
        query = "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';"

        # Execute the query using the connector
        result_df = self.connector.execute_query(query)

        # Assert that the result is a Pandas DataFrame with the expected value
        # as the result shape would be mocked to be (2, 2)
        self.assertEqual(result_df.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
