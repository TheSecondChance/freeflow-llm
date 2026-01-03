# FreeFlow LLM

**Chain multiple free-tier LLM APIs with automatic rate limit fallback.**

FreeFlow LLM is a lightweight Python package that lets you use powerful LLMs completely free by intelligently chaining multiple free-tier providers (Groq, Google Gemini, GitHub Models). When one provider hits a rate limit, it automatically switches to the next one, giving you effectively unlimited free usage!
+`
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
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

# Simple, clean API
response = client.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Ethiopia?"}
    ],
    temperature=0.7,
    max_tokens=100
)

print(response.content)
# Output: "The capital of Ethiopia is Addis Ababa."

print(f"Provider used: {response.provider}")
# Output: "Provider used: groq" (or whichever provider responded)
```

That's it! FreeFlow will automatically try providers in order and handle rate limits transparently.

### Error Handling

```python
from freeflow_llm import FreeFlowClient, NoProvidersAvailableError

client = FreeFlowClient()

try:
    response = client.chat(
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.content)
except NoProvidersAvailableError as e:
    print(f"All providers exhausted: {e}")
    # Handle gracefully or retry later
```

## 🔧 Advanced Configuration

### Default Models

Each provider has a default model configured. When you don't specify a model, FreeFlow uses the provider's default:

```python
from freeflow_llm import FreeFlowClient, config

# Use default models (no model parameter needed)
client = FreeFlowClient()
response = client.chat(messages=[{"role": "user", "content": "Hello!"}])

# Check default models
print(config.DEFAULT_MODELS)
# Output: {
#   'groq': 'llama-3.3-70b-versatile',
#   'gemini': 'gemini-1.5-flash',
#   'mistral': 'mistral-small-latest',
#   ...
# }

# Override with custom model
response = client.chat(
    messages=[{"role": "user", "content": "Hello!"}],
    model="llama-3.1-70b-versatile"  # Override default
)
```

### List Available Providers

```python
client = FreeFlowClient()
print(client.list_providers())
# Output: ['groq', 'gemini', 'github']
```

## 📊 Provider Details

| Provider   | Free Tier Limit | Speed        | Models                 |
| ---------- | --------------- | ------------ | ---------------------- |
| **Groq**   | ~14,400 req/day | ⚡ Very Fast | Llama 3.3 70B, Mixtral |
| **Gemini** | 1,500 req/day   | ⚡ Fast      | Gemini 1.5 Flash       |
| **GitHub** | Rate limited    | ⚡ Fast      | GPT-4o, GPT-4o-mini    |

**Total Effective Limit**: 16,000+ requests/day by chaining all providers! 🎉

## 🛠️ Development

### Setup

```bash
# Clone the repository
git clone https://github.com/thesecondchance/freeflow-llm.git
cd freeflow-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks (recommended)
pre-commit install
```

### Code Quality Standards

We maintain **strict code quality standards**. All PRs must pass:

✅ **Black** - Code formatting  
✅ **Ruff** - Linting & code quality  
✅ **MyPy** - Type checking  
✅ **Circular Import Detection**  
✅ **Import Safety Checks**  
✅ **All Tests**

**Check your code**:

```bash
# Run all checks
python scripts/check_code_quality.py

# Auto-fix what you can
bash scripts/fix_code_quality.sh  # Linux/macOS
powershell scripts/fix_code_quality.ps1  # Windows
```

**Or use pre-commit hooks** (runs automatically on commit):

```bash
pre-commit run --all-files
```

### Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src/freeflow_llm  # With coverage
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ideas for Contributions

- Add more free-tier providers
- Implement streaming support
- Add async support
- Improve error handling
- Write more tests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This package is designed for free-tier usage only. Please respect each provider's rate limits and terms of service. FreeFlow LLM is not affiliated with any of the LLM providers.

## 🙏 Acknowledgments

- **Groq** for their blazingly fast free tier
- **Google** for Gemini API
- **GitHub** for free model access

## 🔗 Links

- [Documentation](https://github.com/thesecondchance/freeflow-llm#readme)
- [PyPI Package](https://pypi.org/project/freeflow-llm/)
- [Issue Tracker](https://github.com/thesecondchance/freeflow-llm/issues)
- [Changelog](https://github.com/thesecondchance/freeflow-llm/blob/main/CHANGELOG.md)

---

**Made with ❤️ for the developer community**

_If this project helps you, please consider giving it a ⭐ on GitHub!_
