import os
import sys
import unittest
from unittest.mock import patch

# Add the absolute path of the root folder to Python path
root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder)

from pipable.llm_client.pipllm import PipLlmApiClient


class TestPipLlmApiClient(unittest.TestCase):
    def setUp(self):
        # Initialize PipLlmApiClient with a mock API base URL for testing
        self.api_base_url = "https://mock-llm-api-url.com"
        self.client = PipLlmApiClient(api_base_url=self.api_base_url)

    @patch("requests.post")
    def test_generate_text(self, mock_post):
        # Mock the requests.post method to return a specific response
        mock_post.return_value.json.return_value = {
            "output": "SELECT first_name FROM actor;"
        }

        # Test the generate_text method
        context = "CREATE TABLE actors (ID INT, first_name TEXT);"
        question = "List first name of all actors."
        generated_query = self.client.generate_text(context, question)

        # Assert that the mocked requests.post method was called with the correct URL and data
        mock_post.assert_called_once_with(
            f"{self.api_base_url}/generate",
            json={"context": context, "question": question},
        )

        # Assert the generated_query matches the expected output from the mock response
        self.assertEqual(generated_query, "SELECT first_name FROM actor;")


if __name__ == "__main__":
    unittest.main()
