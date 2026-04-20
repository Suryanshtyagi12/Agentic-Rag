"""
test_groq.py
------------
Smoke-test for the Groq LLM client.

Usage (from project root with venv active):
    python src/llm/test_groq.py
"""

import sys
import os

# Ensure the project root is on sys.path so `src` imports resolve correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.llm.groq_client import generate_response


def run_test() -> None:
    """Send a simple prompt and print the response."""
    print("=" * 60)
    print("  Groq LLM Smoke Test")
    print("  Model : llama3-70b-8192")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": "You are a concise and helpful AI assistant.",
        },
        {
            "role": "user",
            "content": "What is RAG?",
        },
    ]

    try:
        print("\n[*] Sending prompt: 'What is RAG?'\n")
        response = generate_response(messages)
        print("[✓] Response received:\n")
        print(response)
        print("\n" + "=" * 60)
        print("  Test PASSED")
        print("=" * 60)

    except EnvironmentError as e:
        print(f"\n[✗] Configuration error:\n    {e}")
        print("\nFix: Add your key to .env → GROQ_API_KEY=<your_key>")
        sys.exit(1)

    except RuntimeError as e:
        print(f"\n[✗] API error:\n    {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_test()
