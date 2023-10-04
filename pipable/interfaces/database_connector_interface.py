from abc import ABC, abstractmethod

from pandas import DataFrame


class DatabaseConnectorInterface(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def execute_query(self, query: str) -> DataFrame:
        pass
