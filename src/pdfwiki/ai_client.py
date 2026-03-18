"""
AI client module.
Toggle between Ollama (local/free) and Claude API (cloud/paid) by environment variables.

Task-based model routing:
- "cheap": general use — index, flashcards, cheatsheet (env: PDF_TO_NOTES_MODEL_CHEAP)
- "extract": fact extraction (env: PDF_TO_NOTES_MODEL_EXTRACT)
- "write": high-quality wiki pages and merges (env: PDF_TO_NOTES_MODEL_WRITE)

For Anthropic users: set these env vars to override the defaults.
For Ollama users: task env vars are supported, with OLLAMA_MODEL as fallback.
"""

import os
from dotenv import load_dotenv
import anthropic
from openai import OpenAI

load_dotenv()

# --- CONFIG ---
PROVIDER = os.environ.get("PDF_TO_NOTES_PROVIDER", "anthropic").strip().lower()

# Anthropic API key
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_CLIENT = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# Ollama configuration
OLLAMA_BASE_URL = os.environ.get("PDF_TO_NOTES_OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("PDF_TO_NOTES_OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_CLIENT = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

# Task-based routing: map task names to model selection.
TASK_NAMES = ("cheap", "extract", "write")

# Anthropic API: these env vars specify which Claude model to use for each task.
# If not set, sensible defaults are provided.
TASK_MODELS = {
    "cheap": os.environ.get(
        "PDF_TO_NOTES_MODEL_CHEAP",
        "claude-haiku-4-5-20251001"  # default: fast, low-cost model
    ),
    "extract": os.environ.get(
        "PDF_TO_NOTES_MODEL_EXTRACT",
        "claude-haiku-4-5-20251001"  # default: fast model for structured extraction
    ),
    "write": os.environ.get(
        "PDF_TO_NOTES_MODEL_WRITE",
        "claude-sonnet-4-20250514"  # default: high-quality model for wiki pages
    ),
}

# Ollama task routing (falls back to OLLAMA_MODEL when task-specific vars are absent).
OLLAMA_TASK_MODELS = {
    "cheap": os.environ.get("PDF_TO_NOTES_MODEL_CHEAP", OLLAMA_MODEL),
    "extract": os.environ.get("PDF_TO_NOTES_MODEL_EXTRACT", OLLAMA_MODEL),
    "write": os.environ.get("PDF_TO_NOTES_MODEL_WRITE", OLLAMA_MODEL),
}


def set_provider(provider: str) -> None:
    """Set provider at runtime for the current process."""
    normalized = provider.strip().lower()
    if normalized not in {"anthropic", "ollama"}:
        raise ValueError(
            f"Unsupported provider: {provider}. Expected 'anthropic' or 'ollama'."
        )

    global PROVIDER
    PROVIDER = normalized


def get_provider() -> str:
    """Return the currently active provider."""
    return PROVIDER

def extract_facts(concept: str, context: str, max_tokens: int = 400) -> str:
    """
    Extract structured factual statements about a concept.
    Uses the task="extract" model selection (Haiku or Ollama by default).
    """

    prompt = f"""
    Extract clear, factual statements about "{concept}" from the text below.
    
    Rules:
    - Use bullet points
    - Be concise and precise
    - No explanations, no fluff
    - Avoid repetition
    - Prefer atomic facts (one idea per bullet)
    
    TEXT:
    {context}
    """

    return query(
        prompt=prompt,
        system="You extract factual knowledge for study notes.",
        max_tokens=max_tokens,
        task="extract"
    )


def query(
    prompt: str,
    system: str = "",
    max_tokens: int = 4096,
    task: str = "cheap"
) -> str:
    """
    Send a prompt to the AI and return the response text.

    Tasks:
    - "cheap": general use, cost-optimized (default)
    - "extract": structured fact extraction
    - "write": high-quality, detailed responses (wiki pages, merges)

    For Anthropic: uses task-based model selection from environment variables.
    For Ollama: uses task-based model selection with OLLAMA_MODEL fallback.
    """
    if task not in TASK_NAMES:
        raise ValueError(
            f"Invalid task: {task}. Expected one of: {list(TASK_NAMES)}"
        )

    if PROVIDER == "anthropic":
        model = TASK_MODELS[task]
        return _query_anthropic(prompt, system, max_tokens, model)
    if PROVIDER == "ollama":
        model = OLLAMA_TASK_MODELS.get(task) or OLLAMA_MODEL
        return _query_ollama(prompt, system, max_tokens, model)

    raise ValueError(
        f"Unsupported PDF_TO_NOTES_PROVIDER: {PROVIDER}. "
        "Expected 'anthropic' or 'ollama'."
    )


def _query_anthropic(prompt: str, system: str, max_tokens: int, model: str) -> str:
    if not CLAUDE_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. "
            "Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-..."
        )

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    response = CLAUDE_CLIENT.messages.create(**kwargs)
    return response.content[0].text.strip()


def _query_ollama(prompt: str, system: str, max_tokens: int, model: str) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = OLLAMA_CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    
    if not response.choices[0].message.content:
        return ""
    return response.choices[0].message.content.strip()