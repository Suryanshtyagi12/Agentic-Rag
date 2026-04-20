"""
agent.py
--------
Agentic RAG pipeline: Think → Retrieve → Evaluate → Answer

Loop runs up to MAX_ITERATIONS. On each pass:
  1. THINK     — the LLM analyses the query and extracts key search terms
  2. RETRIEVE  — the retrieval tool fetches relevant chunks
  3. EVALUATE  — the LLM judges whether the context is sufficient
  4. ANSWER    — if sufficient (or max iterations reached), generate final answer

Returns a structured AgentResult for easy consumption by the UI.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.llm.groq_client      import generate_response
from src.retriever.retriever   import Retriever
from src.agent.tools           import retrieval_tool
from src.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    THINK_PROMPT_TEMPLATE,
    EVALUATE_PROMPT_TEMPLATE,
    ANSWER_PROMPT_TEMPLATE,
    FALLBACK_ANSWER,
)

MAX_ITERATIONS = 3
TOP_K_RETRIEVAL = 5


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    query           : str
    answer          : str
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    iterations      : int = 0
    think_output    : str = ""
    evaluate_output : str = ""
    fallback_used   : bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _llm(system: str, user: str) -> str:
    """Thin wrapper: call Groq with a system + user message pair."""
    messages = [
        {"role": "system",  "content": system},
        {"role": "user",    "content": user},
    ]
    return generate_response(messages)


def _extract_search_query(think_output: str, original_query: str) -> str:
    """
    Pull KEY_TERMS from the THINK step output to refine the retrieval query.
    Falls back to the original query if parsing fails.
    """
    for line in think_output.splitlines():
        if line.upper().startswith("KEY_TERMS:"):
            terms = line.split(":", 1)[1].strip()
            if terms:
                return terms
    return original_query


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent(query: str, retriever: Retriever) -> AgentResult:
    """
    Run the Agentic RAG loop for a user query.

    Args:
        query     : User's natural-language question.
        retriever : Loaded Retriever instance (index already built/loaded).

    Returns:
        AgentResult with answer, retrieved chunks, and reasoning trace.
    """
    print("\n" + "=" * 60)
    print("  Agentic RAG — Starting Loop")
    print(f"  Query: {query[:80]}")
    print("=" * 60)

    result = AgentResult(query=query, answer="")
    context_str    : str                  = ""
    retrieved_chunks: List[Dict[str, Any]] = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n── Iteration {iteration}/{MAX_ITERATIONS} ──")
        result.iterations = iteration

        # ── STEP 1: THINK ──────────────────────────────────────────────────
        print("[agent] Step 1: THINK")
        think_prompt = THINK_PROMPT_TEMPLATE.format(query=query)
        think_output = _llm(AGENT_SYSTEM_PROMPT, think_prompt)
        result.think_output = think_output
        print(f"[agent] Think output:\n{think_output[:300]}")

        # Refine search query from think output
        search_query = _extract_search_query(think_output, query)
        print(f"[agent] Refined search query: \"{search_query}\"")

        # ── STEP 2: RETRIEVE ───────────────────────────────────────────────
        print("[agent] Step 2: RETRIEVE")
        try:
            retrieved_chunks = retrieval_tool(search_query, retriever, top_k=TOP_K_RETRIEVAL)
        except RuntimeError as e:
            print(f"[agent] Retrieval failed: {e}")
            result.answer       = FALLBACK_ANSWER
            result.fallback_used = True
            return result

        result.retrieved_chunks = retrieved_chunks
        context_str = retriever.format_context(retrieved_chunks)

        # ── STEP 3: EVALUATE ───────────────────────────────────────────────
        print("[agent] Step 3: EVALUATE")
        eval_prompt    = EVALUATE_PROMPT_TEMPLATE.format(query=query, context=context_str)
        eval_output    = _llm(AGENT_SYSTEM_PROMPT, eval_prompt)
        result.evaluate_output = eval_output
        is_sufficient  = "SUFFICIENT" in eval_output.upper() and "INSUFFICIENT" not in eval_output.upper()
        print(f"[agent] Evaluation: {'✓ SUFFICIENT' if is_sufficient else '✗ INSUFFICIENT'}")

        # ── STEP 4: ANSWER or LOOP ─────────────────────────────────────────
        if is_sufficient or iteration == MAX_ITERATIONS:
            print("[agent] Step 4: ANSWER")
            if not retrieved_chunks:
                result.answer       = FALLBACK_ANSWER
                result.fallback_used = True
            else:
                answer_prompt  = ANSWER_PROMPT_TEMPLATE.format(query=query, context=context_str)
                result.answer  = _llm(AGENT_SYSTEM_PROMPT, answer_prompt)
            break
        else:
            # Refine query for next iteration using missing info from eval
            print("[agent] Context insufficient — refining query for next iteration ...")
            query = f"{query}. Additional focus: {eval_output.replace('INSUFFICIENT', '').strip()}"

    print(f"\n[agent] ✓ Done in {result.iterations} iteration(s)")
    return result
