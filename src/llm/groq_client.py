"""
groq_client.py
--------------
Initializes the Groq LLM client and exposes a clean interface
for generating chat completions using the llama3-70b-8192 model.
"""

import os
from groq import Groq
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env at project root
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "llama3-70b-8192"
MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# Client initialization
# ---------------------------------------------------------------------------
def _get_client() -> Groq:
    """
    Instantiate and return a Groq client.
    Raises EnvironmentError if GROQ_API_KEY is not set.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file: GROQ_API_KEY=your_key_here"
        )
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_response(messages: list) -> str:
    """
    Send a list of chat messages to the Groq API and return the assistant reply.

    Args:
        messages: List of dicts following the OpenAI chat format, e.g.
                  [{"role": "user", "content": "Hello!"}]

    Returns:
        The assistant's response as a plain string.

    Raises:
        EnvironmentError: If GROQ_API_KEY is missing.
        RuntimeError:     If the API call fails for any reason.
    """
    try:
        client = _get_client()
        chat_completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=MAX_TOKENS,
        )
        return chat_completion.choices[0].message.content

    except EnvironmentError:
        # Re-raise config errors as-is so callers can handle them distinctly
        raise

    except Exception as exc:
        raise RuntimeError(
            f"Groq API call failed: {type(exc).__name__} — {exc}"
        ) from exc
