"""STEP 8 · GENERATE  —  the easy part you already know.  (notes: Block 4B)

    PROMPT  ->  >>> THE ANSWER

Nothing here is new. This is the same API-as-a-function call from Week 1 and
the same inference config — temperature, max_tokens — from Week 2. Read the
code and there is genuinely nothing RAG-specific in it.

SAY THIS IN CLASS:
    The model did not change. The context did.

Retrieval is the hard part; generation is the easy part you already know. If
the right chunk was not retrieved, no amount of clever prompting saves the
answer — so spend your effort on search quality, not on the model call.

PROVIDER SWITCH
The default is a local model served by Ollama — no key, no cost, no network.
One line in config.py moves this to a cloud provider. Nothing else in the
project changes — not the chunking, not the embeddings, not the store. That is
your escape hatch if the local model is too slow on a given laptop.
"""

import os

from dotenv import load_dotenv

from config import (LLM_PROVIDER, LLM_PROVIDERS, MAX_TOKENS, TEMPERATURE)

load_dotenv()

if LLM_PROVIDER not in LLM_PROVIDERS:
    raise SystemExit(f"Unknown LLM_PROVIDER {LLM_PROVIDER!r}. "
                     f"Pick one of: {list(LLM_PROVIDERS)}")

_cfg = LLM_PROVIDERS[LLM_PROVIDER]
MODEL = _cfg["model"]


def generate(prompt: str) -> str:
    """Send the grounded prompt to the LLM, return its text answer."""
    # A local provider (Ollama) has no env_var: nothing to look up, no key sent.
    api_key = os.getenv(_cfg["env_var"]) if _cfg["env_var"] else None
    if _cfg["env_var"] and not api_key:
        raise SystemExit(
            f"No {_cfg['env_var']} found (LLM_PROVIDER is '{LLM_PROVIDER}').\n"
            f"  1. get a key at:  {_cfg['signup']}\n"
            f"  2. copy .env.example to .env and add:"
            f"  {_cfg['env_var']}=your-key\n"
            f"\n"
            f"  Or switch provider in config.py to one of "
            f"{[p for p in LLM_PROVIDERS if p != LLM_PROVIDER]}."
        )

    from litellm import completion

    try:
        response = completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=api_key,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        hint = ""
        if LLM_PROVIDER == "ollama":
            hint = (f"\n  Is Ollama running, and is the model pulled?"
                    f"\n    ollama pull {MODEL.split('/', 1)[1]}")
        return f"(generation failed: {type(e).__name__}: {str(e)[:200]}){hint}"
