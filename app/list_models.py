"""
list_models.py
Run this once to see exactly which models YOUR API key can actually use for
generateContent. Model availability shifts over time and varies by account,
so this is more reliable than any hardcoded model name (including the ones
elsewhere in this project).

Usage:
    cd app
    python list_models.py
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

print("Models available to your API key that support generateContent:\n")
for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
    if "generateContent" in actions or not actions:
        print(f"  {m.name}")
