from abc import ABC, abstractmethod
from typing import Any, Generator


class BaseLLMModel(ABC):
    """Abstract base class for LLM models."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: dict) -> None:
        """Initialize the LLM model.

        Parameters
        ----------
        *args : Any
            Positional arguments specific to each model implementation
        **kwargs : dict
            Keyword arguments for model configuration.

        """
        pass

    @abstractmethod
    def get_response(self, messages: Any) -> str:
        """Get a response from the LLM.

        Parameters
        ----------
        messages : list[dict[str, str]]
            A list of message dictionaries

        Returns
        -------
        str
            The LLM's response as a string

        """
        pass

    @abstractmethod
    def get_stream_response(self, messages: Any) -> Generator[str, None, None]:
        """Get a streaming response from the LLM.

        Parameters
        ----------
        messages : Any
            A list of message dictionaries

        Yields
        ------
        str
            Chunks of the response as they arrive

        """
        pass
