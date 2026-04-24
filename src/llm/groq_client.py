"""
groq_client.py
--------------
Initializes the Groq LLM client and exposes a clean interface
for generating chat completions.

CHANGES (v2):
  - Replaced decommissioned model 'llama3-70b-8192' with 'llama-3.1-70b-versatile'
  - Added fallback model 'mixtral-8x7b-32768' if primary fails
  - Added automatic retry logic with clear deprecation messages
  - Model and token config kept as top-level constants (not hardcoded in functions)
"""

import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from the project root (2 levels up from this file)
# Works regardless of where Streamlit sets the CWD
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

# ---------------------------------------------------------------------------
# Model Configuration
# FIX: 'llama3-70b-8192' was decommissioned — replaced with supported models
# ---------------------------------------------------------------------------
MODEL_NAME     = "llama-3.1-70b-versatile"   # Primary (preferred)
FALLBACK_MODEL = "mixtral-8x7b-32768"         # Fallback if primary fails
MAX_TOKENS     = 1024

# Keywords that indicate a model has been deprecated/decommissioned by Groq
_DEPRECATION_KEYWORDS = (
    "decommissioned",
    "deprecated",
    "not supported",
    "model not found",
    "does not exist",
)


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


def _is_deprecation_error(exc: Exception) -> bool:
    """Return True if the exception looks like a model-deprecation error."""
    msg = str(exc).lower()
    return any(kw in msg for kw in _DEPRECATION_KEYWORDS)


def _call_model(client: Groq, model: str, messages: list) -> str:
    """Make a single chat completion call and return the content string."""
    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
    )
    return chat_completion.choices[0].message.content


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_response(messages: list) -> str:
    """
    Send a list of chat messages to the Groq API and return the assistant reply.

    Tries MODEL_NAME first; if that raises a deprecation / bad-request error,
    automatically retries with FALLBACK_MODEL.

    Args:
        messages: List of dicts following the OpenAI chat format, e.g.
                  [{"role": "user", "content": "Hello!"}]

    Returns:
        The assistant's response as a plain string.

    Raises:
        EnvironmentError: If GROQ_API_KEY is missing.
        RuntimeError:     If both primary and fallback API calls fail.
    """
    try:
        client = _get_client()

        # ── Attempt 1: Primary model ────────────────────────────────────────
        try:
            print(f"[groq] Using model: {MODEL_NAME}")
            return _call_model(client, MODEL_NAME, messages)

        except Exception as primary_exc:
            # If it looks like the model was deprecated, warn and retry
            if _is_deprecation_error(primary_exc):
                print(
                    f"[groq] WARNING: Model deprecated or unavailable — "
                    f"'{MODEL_NAME}'. Switching to fallback: '{FALLBACK_MODEL}'"
                )
            else:
                # Non-deprecation error on primary → still try fallback once
                print(
                    f"[groq] Primary model failed ({type(primary_exc).__name__}). "
                    f"Retrying with fallback: '{FALLBACK_MODEL}'"
                )

            # ── Attempt 2: Fallback model ───────────────────────────────────
            try:
                print(f"[groq] Using fallback model: {FALLBACK_MODEL}")
                return _call_model(client, FALLBACK_MODEL, messages)

            except Exception as fallback_exc:
                # Both models failed — raise a clear combined error
                raise RuntimeError(
                    f"Both models failed.\n"
                    f"  Primary  ({MODEL_NAME}): {primary_exc}\n"
                    f"  Fallback ({FALLBACK_MODEL}): {fallback_exc}"
                ) from fallback_exc

    except EnvironmentError:
        # Re-raise config errors as-is so callers can handle them distinctly
        raise

    except RuntimeError:
        raise

    except Exception as exc:
        raise RuntimeError(
            f"Groq API call failed: {type(exc).__name__} — {exc}"
        ) from exc
