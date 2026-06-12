import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()


def call_llm(
    messages: list[ChatCompletionMessageParam],
    schema: dict[str, Any] | None = None,
) -> str:
    """Call MiMo LLM with optional structured output.

    Args:
        messages: Chat messages.
        schema: If provided, forces the model to return JSON matching this schema.
            Should be a dict with 'name', 'strict', and 'schema' keys.

    Returns:
        Model response as a string.
    """
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY not set in environment")
    client = OpenAI(api_key=api_key, base_url="https://api.xiaomimimo.com/v1")

    kwargs: dict[str, Any] = {
        "model": "mimo-v2.5-pro",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 40000,
    }

    if schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": schema,
        }

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    return content or ""
