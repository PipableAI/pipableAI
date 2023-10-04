import os
import sys
import unittest
from unittest.mock import Mock

# Add the absolute path of the root folder to Python path
root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder)

from pipable import Pipable
from pipable.interfaces.database_connector_interface import DatabaseConnectorInterface
from pipable.interfaces.llm_api_client_interface import LlmApiClientInterface


class TestPipable(unittest.TestCase):
    def setUp(self):
        # Mock the LlmApiClientInterface and DatabaseConnectorInterface
        self.mock_llm_api_client = Mock(spec=LlmApiClientInterface)
        self.mock_database_connector = Mock(spec=DatabaseConnectorInterface)

        # Create a Pipable instance with mocked dependencies
        self.pipable = Pipable(
            database_connector=self.mock_database_connector,
            llm_api_client=self.mock_llm_api_client,
        )

    def test_ask_method(self):
        # Set up the mock LlmApiClientInterface's behavior
        context = []
        question = "List all employees."
        generated_sql_query = "SELECT * FROM Employees;"
        self.mock_llm_api_client.generate_text.return_value = generated_sql_query

        # Set up the mock DatabaseConnectorInterface's behavior
        mock_result_df = Mock()
        self.mock_database_connector.execute_query.return_value = mock_result_df

        # Call the ask method
        result = self.pipable.ask(
            question=question,
            table_names=context,
        )

        # Assert that the LlmApiClientInterface's generate_text method was called with the correct arguments
        self.mock_llm_api_client.generate_text.assert_called_once_with(
            context, question
        )

        # Assert that the DatabaseConnectorInterface's execute_query method was called with the generated SQL query
        self.mock_database_connector.execute_query.assert_called_once_with(
            generated_sql_query
        )

        # Assert the result (for demonstration purposes, you might want to modify this assertion based on the actual behavior)
        self.assertIs(result, mock_result_df)


if __name__ == "__main__":
    unittest.main()
