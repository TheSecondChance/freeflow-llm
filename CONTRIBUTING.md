# Contributing to FreeFlow LLM

Thank you for your interest in contributing to FreeFlow LLM! 🎉

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (Python version, OS, provider used)

### Suggesting Features

We welcome feature suggestions! Please open an issue with:

- Clear description of the feature
- Use case / motivation
- Proposed implementation (if you have ideas)

### Pull Requests

1. **Fork the repository**

2. **Clone your fork**

   ```bash
   git clone https://github.com/thesecondchance/freeflow-llm.git
   cd freeflow-llm
   ```

3. **Set up your development environment**

   ```bash
   # Create a virtual environment
   python -m venv venv

   # Activate it
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate

   # Install dependencies
   pip install -e ".[dev]"
   ```

4. **Create a new branch**

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

5. **Make your changes**

   - Write clean, readable code
   - Follow the existing code style
   - Add tests for new features
   - Update documentation as needed

6. **Run tests**

   ```bash
   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=freeflow_llm

   # Run specific test file
   pytest tests/test_client.py
   ```

7. **Commit your changes**

   ```bash
   git add .
   git commit -m "feat: add new feature" # or "fix: resolve bug"
   ```

   Follow conventional commit format:

   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `test:` - Adding or updating tests
   - `refactor:` - Code refactoring
   - `chore:` - Maintenance tasks

8. **Push to your fork**

   ```bash
   git push origin feature/your-feature-name
   ```

9. **Open a Pull Request**

   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template with details about your changes

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write descriptive variable and function names
- Keep functions focused and small
- Add docstrings to public functions and classes

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage (>80%)
- Test edge cases and error conditions

### Documentation

- Update README.md if adding new features
- Add docstrings to new functions/classes
- Update examples if API changes
- Keep documentation clear and concise

## Code Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged
4. Your contribution will be credited in the release notes

## Questions?

Feel free to open an issue for any questions or join our discussions!

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).
