# Contributing to Spoof Detection

Thank you for your interest in contributing! This guide will help you get started.

## Getting Started

1. **Fork the repository** and clone your fork locally.
2. **Create a virtual environment** and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   pip install -r requirements-dev.txt
   ```

3. **Create a branch** for your changes:

   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Code Style

This project uses the following tools for code quality:

- **[Black](https://black.readthedocs.io/)** for code formatting
- **[isort](https://pycqa.github.io/isort/)** for import sorting
- **[flake8](https://flake8.pycqa.org/)** for linting

Format your code before committing:

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/ --max-line-length=100
```

### Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=src/spoof_detection --cov-report=term-missing
```

## Submitting Changes

1. Ensure all tests pass and code is formatted.
2. Commit your changes with a clear, descriptive message.
3. Push your branch and open a Pull Request.
4. Describe your changes in the PR description.

## Reporting Issues

- Use the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md) template for bugs.
- Use the [Feature Request](.github/ISSUE_TEMPLATE/feature_request.md) template for suggestions.

## Code of Conduct

Please be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive experience for everyone.
