"""
Spike B — LLM provider evaluation (free tiers, Georgian quality, tool-calling).

Providers to test:
  - Google Gemini (gemini-1.5-flash or current free default — VERIFY on ai.google.dev)
  - Groq (llama-3.1-8b-instant or current free default — VERIFY on console.groq.com)

What to verify BEFORE running:
  1. Current free-tier request/token limits for each provider.
     Source URLs must be recorded in DECISIONS.md.
     If the limit is not on an official page, mark UNVERIFIED.
  2. Whether each provider supports structured tool-calling with JSON schemas.

What this measures (on the same 20 Georgian questions):
  - Answer quality: does it understand the question in Georgian?
  - Does it answer in Georgian when asked in Georgian?
  - Does it produce valid JSON when asked with a JSON schema?
  - Latency per call (seconds)

Results are printed to stdout.  Copy the output into docs/DECISIONS.md.
"""

# TODO: implement after the owner has API keys in .env
raise NotImplementedError("Spike B not yet implemented — see docstring for instructions.")
