import json
import os
from typing import Any

import dotenv


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self._llm_api_key = os.getenv("LLM_API_KEY")
        self._llm_model_name = None  # Model name set via UI only

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        dotenv.load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Parameters
        ----------
        file_path : str
            Path to the JSON configuration file.

        Returns
        -------
        dict[str, Any]
            Dict containing server configuration.

        Raises
        ------
        FileNotFoundError
            If configuration file doesn't exist.
        JSONDecodeError
            If configuration file is invalid JSON.

        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.

        Returns
        -------
        str
            The API key as a string.

        Raises
        ------
        ValueError
            If the API key is not found in environment variables.

        """
        if not self._llm_api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self._llm_api_key

    @property
    def llm_model_name(self) -> str:
        """Get the LLM model name.

        Returns
        -------
        str
            The model name as a string.

        Raises
        ------
        ValueError
            If the model name is not configured through the UI agent configuration.

        """
        if not self._llm_model_name:
            raise ValueError("Model name not configured. Please configure your agent through the UI.")
        return self._llm_model_name
