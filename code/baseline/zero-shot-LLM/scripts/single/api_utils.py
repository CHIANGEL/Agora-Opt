import os
import time
from typing import Any, Optional
import openai


def get_api_base_url() -> str:
    api_url = os.getenv("API_URL")
    if not api_url:
        raise RuntimeError("Missing API_URL environment variable")
    return api_url


def get_api_key() -> str:
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Missing API_KEY environment variable")
    return api_key


def get_response(prompt: str, model: str, max_try: int = 5, temperature: float = 0.0) -> str:
    """Get response from API with retry logic."""
    client = openai.OpenAI(
        api_key=get_api_key(),
        base_url=get_api_base_url(),
    )
    for i in range(max_try):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Attempt {i + 1} failed with error: {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed to get response from model {model} after {max_try} attempts.")


def check_model_available(model: str) -> bool:
    """Check if a model is available in the API."""
    try:
        client = openai.OpenAI(
            api_key=get_api_key(),
            base_url=get_api_base_url(),
        )
        client.models.retrieve(model)
        return True
    except openai.NotFoundError:
        return False
    except Exception as e:
        print(f"An error occurred while checking model availability: {e}")
        return False
