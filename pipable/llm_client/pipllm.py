import json

import requests

from pipable.core.dev_logger import dev_logger
from pipable.interfaces.llm_api_client_interface import LlmApiClientInterface


class PipLlmApiClient(LlmApiClientInterface):
    """A client class for interacting with the Pipable Language Model API.

    This class provides methods to communicate with a language model API to generate SQL queries
    based on contextual information and user queries. It facilitates sending requests to the API
    and receiving generated SQL queries as responses.

    Args:
        api_base_url (str): The base URL of the Language Model API.

    Attributes:
        api_base_url (str): The base URL of the Language Model API.

    Example:
        To use this client, create an instance of `PipLlmApiClient`, configure it with the API base URL,
        and use the `generate_text` method to generate SQL queries.

        .. code-block:: python

            from pipable.pip_llm_api_client import PipLlmApiClient

            # Create a PipLlmApiClient instance
            llm_api_client = PipLlmApiClient(api_base_url="https://your-llm-api-url.com")

            # Generate an SQL query based on context and user query
            context = "CREATE TABLE Employees (ID INT, NAME TEXT);"
            user_query = "List all employees."
            generated_query = llm_api_client.generate_text(context, user_query)

    Methods:
        - generate_text(context: str, question: str) -> str: Generate an SQL query based on context and user query.

    Raises:
        requests.exceptions.RequestException: If there is an issue with the API request.
    """

    def __init__(self, api_base_url: str):
        """Initialize a PipLlmApiClient instance.

        Args:
            api_base_url (str): The base URL of the Language Model API.
        """
        self.api_base_url = api_base_url

    def generate_text(self, context: str, question: str) -> str:
        """Generate an SQL query based on contextual information and user query.

        Args:
            context (str): The context or CREATE TABLE statements for the query.
            question (str): The user's query in simple English.

        Returns:
            str: The generated SQL query.

        Raises:
            requests.exceptions.RequestException: If there is an issue with the API request.
        """
        endpoint = "/generate"
        url = self.api_base_url + endpoint
        data = {"context": context, "question": question}
        response = self._make_post_request(url, data)
        return response.get("output")

    def _make_post_request(self, url, data):
        """Make a POST request to the specified URL with the provided data.

        Args:
            url (str): The URL to make the POST request to.
            data (dict): The data to send with the POST request.

        Returns:
            dict: The JSON response from the API.

        Raises:
            requests.exceptions.RequestException: If there is an issue with the API request.
        """

        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error making POST request: {str(e)}")


__all__ = ["PipLlmApiClient"]
