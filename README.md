# FreeFlow LLM

**Chain multiple free-tier LLM APIs with automatic rate limit fallback.**

FreeFlow LLM is a lightweight Python package that lets you use powerful LLMs completely free by intelligently chaining multiple free-tier providers (currently **Groq** and **Google Gemini**, with more on the way). When one provider hits a rate limit, it automatically switches to the next one, giving you effectively unlimited free usage.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## ✨ Features

- **100% Free-Tier Only**: No paid tiers, no credit card required
- **Automatic Fallback**: Detects rate limits (HTTP 429) and switches providers instantly
- **Multiple API Keys**: Rotate through several keys per provider to multiply your limits
- **Smart Prioritization**: Starts with the fastest providers (Groq), falls back to others
- **Clean & Unified API**: Simple, consistent `client.chat()` interface across all providers
- **16,000+ Requests/Day**: Aggregate free usage across all providers

## Quick Start

### Installation

```bash
pip install freeflow-llm
```

### Set Up API Keys

Get free API keys from these providers (you only need at least one):

1. **Groq** (Recommended): https://console.groq.com/keys
2. **Google Gemini**: https://makersuite.google.com/app/apikey

Set them as environment variables (or place them in a `.env` file):

```bash
export GROQ_API_KEY="your_groq_key"
export GEMINI_API_KEY="your_gemini_key"
```

#### Multiple API Keys per Provider

Configure **multiple keys** per provider. When a rate limit is hit, FreeFlow rotates through all available keys before moving to the next provider:

```bash
# JSON array (recommended for complex keys)
export GEMINI_API_KEY='["key1", "key2", "key3"]'

# Comma-separated (simpler)
export GROQ_API_KEY="groq_key1,groq_key2"

# Single key (traditional)
export GEMINI_API_KEY="single_key"
```

More keys means a higher effective limit. For example, 3 Gemini keys = 4,500 requests/day instead of 1,500.

### Basic Usage

```python
from freeflow_llm import FreeFlowClient

# Context manager ensures HTTP connections are cleaned up (recommended)
with FreeFlowClient() as client:
    response = client.chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Ethiopia?"},
        ],
        temperature=0.7,
        max_tokens=100,
    )

    print(response.content)       # "The capital of Ethiopia is Addis Ababa."
    print(response.provider)      # "groq" (or whichever provider responded)
```

FreeFlow automatically tries providers in order and handles rate limits transparently.

### Error Handling

```python
from freeflow_llm import FreeFlowClient, NoProvidersAvailableError

with FreeFlowClient() as client:
    try:
        response = client.chat(messages=[{"role": "user", "content": "Hello!"}])
        print(response.content)
    except NoProvidersAvailableError as e:
        print(f"All providers exhausted: {e}")
```

## 🔧 Advanced Configuration

```python
from freeflow_llm import FreeFlowClient, config
from freeflow_llm.providers import GeminiProvider, GroqProvider

# Check default models (used when no `model` is specified)
print(config.DEFAULT_MODELS)
# {'groq': 'llama-3.3-70b-versatile', 'gemini': 'gemini-2.5-flash', ...}

# Override the default model per call
with FreeFlowClient() as client:
    client.chat(
        messages=[{"role": "user", "content": "Hello!"}],
        model="gemini-2.5-flash",
    )

# Configure providers manually with multiple keys
custom_providers = [
    GroqProvider(api_key=["key1", "key2"]),
    GeminiProvider(api_key=["key1", "key2", "key3"]),
]
client = FreeFlowClient(providers=custom_providers)
print(client.list_providers())  # ['groq', 'gemini']
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

To set up for development:

```bash
git clone https://github.com/thesecondchance/freeflow-llm.git
cd freeflow-llm
pip install -e ".[dev]"
pre-commit install
pytest tests/ -v
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This package is designed for free-tier usage only. Please respect each provider's rate limits and terms of service. FreeFlow LLM is not affiliated with any of the LLM providers.

## 🔗 Links

- [PyPI Package](https://pypi.org/project/freeflow-llm/)
- [Source Code](https://github.com/thesecondchance/freeflow-llm)
- [Issue Tracker](https://github.com/thesecondchance/freeflow-llm/issues)

---

**Made with ❤️ for the developer community.** If this project helps you, please consider giving it a ⭐ on GitHub!
