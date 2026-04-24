"""
groq_client.py
--------------
Groq LLM client with dynamic model selection, automatic fallback,
and fail-safe error handling.

Design principles:
  - Model names are NEVER hardcoded inside functions — only in MODEL_CONFIG
  - Primary model is tried first; fallback is used automatically on deprecation
  - If BOTH models fail, a safe string is returned (no crash / RuntimeError)
  - Model can be overridden at runtime via GROQ_MODEL env variable
  - Full debug logging at every decision point
"""

import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from the project root (2 levels up from this file)
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

# ---------------------------------------------------------------------------
# Centralised Model Configuration
# To change models: edit ONLY this dict — nowhere else.
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    "primary" : "llama-3.1-8b-instant",  # Fast, currently active on Groq
    "fallback" : "gemma2-9b-it",          # Stable Google Gemma 2 via Groq
}

# Allow runtime override via environment variable:
#   GROQ_MODEL=llama-3.1-8b-instant  (in .env or shell)
_env_override = os.getenv("GROQ_MODEL", "").strip()
if _env_override:
    MODEL_CONFIG["primary"] = _env_override
    print(f"[groq] Env override: GROQ_MODEL={_env_override}")

# Convenience alias used externally (e.g. app/main.py badge)
MODEL_NAME = MODEL_CONFIG["primary"]

MAX_TOKENS = 1024

# Error conditions that indicate a model is no longer available
_DEPRECATION_SIGNALS = (
    "decommissioned",
    "deprecated",
    "model_decommissioned",
    "invalid_request_error",
    "model not found",
    "does not exist",
    "not supported",
)

# Message returned to the caller when ALL models fail (never crashes the app)
UNAVAILABLE_MSG = (
    "LLM service temporarily unavailable. Please try again later."
)


# ---------------------------------------------------------------------------
# Internal helpers
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
    """Return True if the exception signals a deprecated / decommissioned model."""
    msg = str(exc).lower()
    return any(signal in msg for signal in _DEPRECATION_SIGNALS)


def _call_model(client: Groq, model: str, messages: list) -> str:
    """Execute one chat-completion call and return the content string."""
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
    )
    return completion.choices[0].message.content


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_response(messages: list) -> str:
    """
    Send messages to the Groq API and return the assistant reply.

    Execution order:
      1. Try MODEL_CONFIG["primary"]  (or GROQ_MODEL env override)
      2. On any deprecation/invalid error → retry with MODEL_CONFIG["fallback"]
      3. If fallback also fails         → return UNAVAILABLE_MSG (no crash)

    Args:
        messages: List of dicts: [{"role": "user", "content": "..."}]

    Returns:
        Assistant reply string, or UNAVAILABLE_MSG if all models fail.

    Raises:
        EnvironmentError: Only if GROQ_API_KEY is missing.
    """
    primary  = MODEL_CONFIG["primary"]
    fallback = MODEL_CONFIG["fallback"]

    try:
        client = _get_client()
    except EnvironmentError:
        raise  # Config error — surface immediately, don't swallow

    # ── Attempt 1: Primary model ─────────────────────────────────────────────
    try:
        print(f"[groq] Trying primary model: {primary}")
        response = _call_model(client, primary, messages)
        print(f"[groq] Success with primary model: {primary}")
        return response

    except Exception as primary_exc:
        short_err = str(primary_exc)[:200]
        if _is_deprecation_error(primary_exc):
            print(
                f"[groq] PRIMARY MODEL DECOMMISSIONED: '{primary}'\n"
                f"        Error: {short_err}\n"
                f"        → Switching to fallback: '{fallback}'"
            )
        else:
            print(
                f"[groq] Primary model failed ({type(primary_exc).__name__}): {short_err}\n"
                f"        → Retrying with fallback: '{fallback}'"
            )

    # ── Attempt 2: Fallback model ────────────────────────────────────────────
    try:
        print(f"[groq] Trying fallback model: {fallback}")
        response = _call_model(client, fallback, messages)
        print(f"[groq] Success with fallback model: {fallback}")
        return response

    except Exception as fallback_exc:
        short_err = str(fallback_exc)[:200]
        print(
            f"[groq] FALLBACK MODEL ALSO FAILED: '{fallback}'\n"
            f"        Error: {short_err}\n"
            f"        → Returning safe unavailable message."
        )

    # ── Fail-safe: both models failed ───────────────────────────────────────
    return UNAVAILABLE_MSG
