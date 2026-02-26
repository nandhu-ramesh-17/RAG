from abc import ABC, abstractmethod
from typing import List

class BaseEmbedding(ABC):
    """
    Abstract base class for embedding providers.
    Any embedding implementation (HuggingFace, OpenAI, etc.)
    must inherit from this class.
    """

    @abstractmethod
    async def embed(self, text : str) -> List[float]:
        """
        Function for embedding single text
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts : List[str]) -> List[List[float]]:
        """
        Embedding Generation for a batch of text
        """
        pass