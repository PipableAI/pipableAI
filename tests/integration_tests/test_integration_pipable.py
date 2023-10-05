# This Test is specifc to a local postgressDB instance, with the databse name of
#  sampleDB, and the data as from the DVD Rental Database, downloaded from
#  https://www.postgresqltutorial.com/postgresql-getting-started/postgresql-sample-database/

import os
import sys
import unittest
from unittest.mock import Mock, patch

# Add the absolute path of the root folder to Python path
root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_folder)

from pipable import Pipable
from pipable.core.postgresql_connector import PostgresConfig, PostgresConnector
from pipable.interfaces.llm_api_client_interface import LlmApiClientInterface


class MockLlmApiClient(LlmApiClientInterface):
    # Mock LlmApiClientInterface for pipable API
    pass


class IntegrationTestPipable(unittest.TestCase):
    def setUp(self):
        # Define PostgreSQL configuration for your local database
        postgres_config = PostgresConfig(
            host="localhost",
            port=5432,  # Replace with your port number
            database="sampleDB",
            user="postgres",
            password="postgres",
        )

        # Create a real PostgresConnector instance for connecting to the database
        self.local_database_connector = PostgresConnector(postgres_config)

    def tearDown(self):
        self.local_database_connector.disconnect()

    @patch("pipable.llm_client.pipllm.PipLlmApiClient")  # Mock the LLM API client
    def test_ask_and_execute_method(self, mock_llm_api_client):
        # Arrange
        # Set up the mock LLM API client
        mock_llm_instance = mock_llm_api_client.return_value
        mock_llm_instance.generate_text.return_value = "SELECT first_name FROM actor;"

        # Act
        # Initialize Pipable with mocked dependencies
        pipable = Pipable(
            database_connector=self.local_database_connector,
            llm_api_client=mock_llm_instance,
        )

        # Call the 'ask_and_execute' method
        result = pipable.ask_and_execute(
            table_names=["actor", "city"],
            question="List first name of all actors.",
        )

        # Assert
        # Ensure the LLM API client's 'generate_text' method was called with the correct arguments
        mock_llm_instance.generate_text.assert_called_once_with(
            "CREATE TABLE actor (actor_id integer, last_update timestamp without time zone, first_name character varying, last_name character varying); CREATE TABLE city (last_update timestamp without time zone, city_id integer, country_id smallint, city character varying);",
            "List first name of all actors.",
        )

    @patch("pipable.llm_client.pipllm.PipLlmApiClient")  # Mock the LLM API client
    def test_ask_method(self, mock_llm_api_client):
        # Arrange
        # Set up the mock LLM API client
        mock_llm_instance = mock_llm_api_client.return_value
        mock_llm_instance.generate_text.return_value = "SELECT first_name FROM actor;"

        # Act
        # Initialize Pipable with mocked dependencies
        pipable = Pipable(
            database_connector=self.local_database_connector,
            llm_api_client=mock_llm_instance,
        )

        # Call the 'ask_and_execute' method
        result = pipable.ask(
            table_names=["actor", "city"],
            question="List first name of all actors.",
        )

        # Assert
        # Ensure the LLM API client's 'generate_text' method was called with the correct arguments
        mock_llm_instance.generate_text.assert_called_once_with(
            "CREATE TABLE actor (actor_id integer, last_update timestamp without time zone, first_name character varying, last_name character varying); CREATE TABLE city (last_update timestamp without time zone, city_id integer, country_id smallint, city character varying);",
            "List first name of all actors.",
        )

        self.assertEqual(result, "SELECT first_name FROM actor;")


if __name__ == "__main__":
    unittest.main()
