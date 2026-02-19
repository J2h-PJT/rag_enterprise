from abc import ABC, abstractmethod

class BaseLLM(ABC):

    @abstractmethod
    def get_model(self):
        pass
