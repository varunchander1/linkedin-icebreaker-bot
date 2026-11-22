
"""Module for interfacing with IBM watsonx.ai embeddings and OpenAI LLMs."""

import os
import logging
from typing import Dict, Any, Optional


from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

import config

logger = logging.getLogger(__name__)

def create_embedding() -> OpenAIEmbedding:
    """Creates an OpenAI Embedding model for vector representation.

    Reads the API key from the OPENAI_API_KEY environment variable.

    Returns:
        OpenAIEmbedding model.
    """
    embedding = OpenAIEmbedding(
        model=config.EMBEDDING_MODEL_ID,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    logger.info(f"Created OpenAI Embedding model: {config.EMBEDDING_MODEL_ID}")
    return embedding



def create_llm(
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS
) -> OpenAILike:
    """Creates an OpenAI chat LLM for generating responses.

    Reads the API key from the OPENAI_API_KEY environment variable.

    Args:
        temperature: Temperature for controlling randomness in generation (0.0 to 1.0).
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        OpenAILike chat model.
    """
    # GPT-5 family models only accept the default temperature of 1
    if config.LLM_MODEL_ID.startswith("gpt-5"):
        temperature = 1.0

    # OpenAILike is used instead of llama_index's OpenAI class because the pinned
    # version does not recognise newer model names such as gpt-5.4-mini
    llm = OpenAILike(
        model=config.LLM_MODEL_ID,
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=config.OPENAI_API_BASE,
        is_chat_model=True,
        context_window=config.CONTEXT_WINDOW,
        temperature=temperature,
        # GPT-5 models reject `max_tokens`; `max_completion_tokens` works for all current chat models
        max_tokens=None,
        additional_kwargs={"max_completion_tokens": max_new_tokens},
    )
    logger.info(f"Created OpenAI LLM model: {config.LLM_MODEL_ID}")
    return llm

def change_llm_model(new_model_id: str) -> None:
    """Change the LLM model to use.
    
    Args:
        new_model_id: New LLM model ID to use.
    """
    global config
    config.LLM_MODEL_ID = new_model_id
    logger.info(f"Changed LLM model to: {new_model_id}")