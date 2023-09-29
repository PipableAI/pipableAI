"""Pipable Client

This module provides a client interface for interacting with the Pipable package.
It allows users to send queries to a language model and get the response.

Classes:
    - PipableClient: A client class for querying the language model and executing SQL queries on the server.

Example:
    To use this client, create an instance of `PipableClient`, configure it with the necessary parameters,
    and use the `ask` method to send queries and retrieve results.

    Example:
    ```
    python
    from pipable.client import PipableClient

    # Create a PipableClient instance
    client = PipableClient(llm_api_url="https://your-llm-api-url.com")

    # Send a query to the language model and execute it on the server
    result_df = client.ask(context="CREATE TABLE Employees (ID INT, NAME TEXT);", question="List all employees.")
    ```
"""

import requests
import json

class PipLlmApiClient:
    """A client class for interacting with the Language Model API.

    This class provides methods to communicate with a language model API to generate SQL queries
    based on contextual information and user queries. It facilitates sending requests to the API
    and receiving generated SQL queries as responses.

    Args:
        api_base_url (str): The base URL of the Language Model API.

    Example:
        To use this client, create an instance of `PipLlmApiClient`, configure it with the API base URL,
        and use the `generate_query` method to generate SQL queries.

        Example:
        ```python
        from pipable.pip_llm_api_client import PipLlmApiClient

        # Create a PipLlmApiClient instance
        llm_api_client = PipLlmApiClient(api_base_url="https://your-llm-api-url.com")

        # Generate an SQL query based on context and user query
        context = "CREATE TABLE Employees (ID INT, NAME TEXT);"
        user_query = "List all employees."
        generated_query = llm_api_client.generate_query(context, user_query)
        ```
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
            user_query (str): The user's query in simple English.

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

    def train_llm(self, dataset_path):
        endpoint = "/train"
        url = self.api_base_url + endpoint
        data = {"dataset_path": dataset_path}
        response = self._make_post_request(url, data)
        return response

    def _make_post_request(self, url, data):
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error making POST request: {str(e)}")

# Example usage:
if __name__ == "__main__":
    api_base_url = "https://your-llm-api-url.com"
    client = PipLlmApiClient(api_base_url)

    # Example 1: Generate text from LLM
    context = "<DETAILS ABOUT TABLE>"
    question = "<QUERY TO PERFORM IN SIMPLE ENGLISH>"
    generated_text = client.generate_text(context, question)
    print("Generated Text:", generated_text)

    # Example 2: Fine-tune LLM
    dataset_path = "<PATH TO DATASET.JSON>"
    response = client.train_llm(dataset_path)
    print("Training Status:", response.get("status"))
    print("Message:", response.get("message"))

__all__ = ["PipLlmApiClient"]
