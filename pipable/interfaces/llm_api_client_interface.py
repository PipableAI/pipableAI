from abc import ABC, abstractmethod


class LlmApiClientInterface(ABC):
    """Abstract base class for Language Model API client interfaces.

    This class defines the interface for interacting with a Language Model API.
    Concrete implementations must inherit from this class and provide implementations for the abstract methods.

    Attributes:
        None

    Methods:
        - generate_text(context: str, question: str) -> str: Generate text based on the given context and question.

    Example:
        To create a custom API client for a specific language model, inherit from this class and provide implementations
        for the abstract methods.

        .. code-block:: python

            from abc import ABC, abstractmethod

            class CustomLlmApiClient(LlmApiClientInterface):
                def __init__(self, api_key):
                    # Initialize the API client with the provided API key
                    pass

                def generate_text(self, context: str, question: str) -> str:
                    # Implement logic to generate text based on the given context and question
                    pass
    """

    @abstractmethod
    def generate_text(self, context: str, question: str) -> str:
        """Generate text based on the given context and question.

        Args:
            context (str): The context for text generation.
            question (str): The question to be answered in the generated text.

        Returns:
            str: The generated text.
        """
        pass


__all__ = ["LlmApiClientInterface"]
