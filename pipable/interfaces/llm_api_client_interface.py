from abc import ABC, abstractmethod


class LlmApiClientInterface(ABC):
    @abstractmethod
    def generate_text(self, context: str, question: str) -> str:
        pass
