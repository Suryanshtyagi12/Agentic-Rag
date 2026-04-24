"""
test_groq.py
------------
Smoke-test for the Groq LLM client.

Tests:
  1. Simple prompt → ensure response returns
  2. Print which model was actually used
  3. Verify UNAVAILABLE_MSG is returned (not a crash) if models fail

Usage (from project root with venv active):
    python src/llm/test_groq.py
"""

import sys
import os

# Ensure the project root is on sys.path so `src` imports resolve correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.llm.groq_client import generate_response, MODEL_CONFIG, UNAVAILABLE_MSG


def run_test() -> None:
    """Send a simple prompt and print the response."""
    primary  = MODEL_CONFIG["primary"]
    fallback = MODEL_CONFIG["fallback"]

    print("=" * 60)
    print("  Groq LLM Smoke Test")
    print(f"  Primary model  : {primary}")
    print(f"  Fallback model : {fallback}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": "You are a concise and helpful AI assistant.",
        },
        {
            "role": "user",
            "content": "What is RAG (Retrieval Augmented Generation)? Answer in 2-3 sentences.",
        },
    ]

    print("\n[*] Sending prompt ...\n")
    response = generate_response(messages)

    if response == UNAVAILABLE_MSG:
        print(f"\n[[WARN]] LLM unavailable — both models failed.")
        print(f"         Response: {response}")
        sys.exit(1)
    else:
        print("[[OK]] Response received:\n")
        print(response)
        print("\n" + "=" * 60)
        print("  Test PASSED")
        print("=" * 60)


if __name__ == "__main__":
    try:
        run_test()
    except EnvironmentError as e:
        print(f"\n[[FAIL]] Configuration error:\n    {e}")
        print("\nFix: Add your key to .env -> GROQ_API_KEY=<your_key>")
        sys.exit(1)
