# FreeFlow LLM

**Chain multiple free-tier LLM APIs with automatic rate limit fallback.**

FreeFlow LLM is a lightweight Python package that lets you use powerful LLMs completely free by intelligently chaining multiple free-tier providers (Groq, Google Gemini, GitHub Models). When one provider hits a rate limit, it automatically switches to the next one, giving you effectively unlimited free usage!
+`
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://github.com/python/mypy)

## ✨ Features

- **100% Free-Tier Only** — No paid tiers, no credit card required
- **Automatic Fallback** — Detects rate limits (HTTP 429) and switches providers instantly
- **Smart Prioritization** — Starts with fastest providers (Groq), falls back to others
- **OpenAI-Compatible API** — Familiar `client.chat.completions.create()` interface
- **Lightweight & Simple** — Minimal dependencies, easy to use
- **16,000+ Requests/Day** — Aggregate free usage across all providers

## Quick Start

### Installation

```bash
pip install freeflow-llm
```

### Set Up API Keys

Get free API keys from these providers (you only need at least one):

1. **Groq** (Recommended): https://console.groq.com/keys
2. **Google Gemini**: https://makersuite.google.com/app/apikey
3. **GitHub Models**: https://github.com/settings/tokens

Set them as environment variables:

```bash
export GROQ_API_KEY="your_groq_key"
export GOOGLE_API_KEY="your_google_key"
export GITHUB_TOKEN="your_github_token"
```

Or create a `.env` file:

```env
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
GITHUB_TOKEN=your_github_token
```

### Basic Usage

```python
from freeflow_llm import FreeFlowClient

# Initialize client (auto-loads API keys from environment)
client = FreeFlowClient()

# Use familiar OpenAI-style API
response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Ethiopia?"}
    ],
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message.content)
# Output: "The capital of Ethiopia is Addis Ababa."

print(f"Provider used: {response.provider}")
# Output: "Provider used: groq" (or whichever provider responded)
```

That's it! FreeFlow will automatically try providers in order and handle rate limits transparently.
