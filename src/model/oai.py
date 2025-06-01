from typing import Generator

import dotenv
from openai import OpenAI
from openai.types.responses import ResponseTextDeltaEvent

from src.model.base import BaseLLMModel

dotenv.load_dotenv()


class OpenAIClient(BaseLLMModel):
    """OpenAI client implementation for LLM interactions."""

    def __init__(self, model_name: str, api_key: str) -> None:
        """Initialize the OpenAI client.

        Parameters
        ----------
        model_name : str
            The name of the OpenAI model to use (e.g., 'gpt-4o-mini')
        api_key : str
            The OpenAI API key for authentication

        """
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def get_response(self, messages: list[dict[str, str]]) -> str:
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
        completion = self.client.responses.create(
            model=self.model_name,
            input=messages,  # type: ignore
        )
        return completion.output_text

    def get_stream_response(self, messages: list[dict[str, str]]) -> Generator[str, None, None]:
        """Get a streaming response from the LLM.

        Parameters
        ----------
        messages : list[dict[str, str]]
            A list of message dictionaries

        Yields
        ------
        str
            Chunks of the response as they arrive

        """
        stream = self.client.responses.create(
            model=self.model_name,
            input=messages,  # type: ignore
            stream=True,
        )

        for chunk in stream:
            if isinstance(chunk, ResponseTextDeltaEvent):
                yield chunk.delta
            else:
                continue


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    client = OpenAIClient(
        model_name="gpt-4o-mini",
        api_key=api_key,
    )
    # Testing response
    print(client.get_response([{"role": "user", "content": "Who are you?"}]))

    # Testing stream response
    for chunk in client.get_stream_response([{"role": "user", "content": "Who are you?"}]):
        print(chunk, end="", flush=True)
