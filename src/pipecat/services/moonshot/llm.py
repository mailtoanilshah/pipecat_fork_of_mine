#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Moonshot AI LLM Service implementation using OpenAI-compatible interface."""

from loguru import logger

from pipecat.services.openai.llm import OpenAILLMService


class MoonshotLLMService(OpenAILLMService):
    """A service for interacting with Moonshot AI's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Moonshot AI's native API endpoint,
    which provides better tool calling support than accessing Moonshot through Groq.
    
    Moonshot AI (Kimi) provides OpenAI-compatible API with proper tool calling support.
    Get your API key from: https://platform.moonshot.cn/
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.moonshot.ai/v1",
        model: str = "moonshot-v1-8k",
        params: OpenAILLMService.InputParams = OpenAILLMService.InputParams(),
    ):
        """Initialize Moonshot LLM service.

        Args:
            api_key: The API key for accessing Moonshot AI's API.
            base_url: The base URL for Moonshot API. Defaults to "https://api.moonshot.ai/v1".
            model: The model identifier to use. Defaults to "moonshot-v1-8k".
                Available models:
                - moonshot-v1-8k: 8K context window
                - moonshot-v1-32k: 32K context window
                - moonshot-v1-128k: 128K context window
            params: Input parameters for the LLM service.
        """
        super().__init__(api_key=api_key, base_url=base_url, model=model, params=params)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Moonshot API endpoint.

        Args:
            api_key: API key for authentication. If None, uses instance api_key.
            base_url: Base URL for the API. If None, uses instance base_url.
            **kwargs: Additional arguments passed to the client constructor.

        Returns:
            An OpenAI-compatible client configured for Moonshot AI's API.
        """
        logger.debug(f"Creating Moonshot client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
