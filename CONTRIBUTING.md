# Contributing to FINRIX

Thank you for considering contributing to FINRIX, an autonomous dual-path quantitative trading terminal for XAUUSD macro-forecasting. We welcome contributions from the quantitative finance and AI systems communities.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Testing Standards](#testing-standards)
- [Security Guidelines](#security-guidelines)
- [Style Guidelines](#style-guidelines)
- [License](#license)

## Code of Conduct
This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites
- Python 3.10+
- Git
- Modal account (optional, for GPU pipeline)

### Clone the Repository
```bash
git clone https://github.com/carnoba/trading-prediction.git
cd trading-prediction
pip install -r requirements.txt
```

### Running Locally
```bash
python app.py
# Or use the CLI shortcut:
finrix
```

## Development Workflow

1. **Fork the repository** on GitHub
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the style guidelines below
4. **Run the test suite**:
   ```bash
   python -m pytest -q
   python -m compileall -q app.py macro_forecast_engine
   ```
5. **Commit your changes** with a descriptive message:
   ```bash
   git commit -m "feat: description of your change"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request** against `main`

## Testing Standards

All contributions must pass:
- `python -m pytest -q` — full test suite (207 tests)
- `python -m compileall -q app.py macro_forecast_engine` — compilation check

## Security Guidelines

FINRIX handles financial data and execution logic. Contributors must:
- Never commit API secrets, keys, or credentials
- Use environment variables or `.env` files for sensitive data
- Validate all external inputs (SQL injection, XSS, NaN/Inf guards)
- Implement fallback mechanisms for external API failures
- Ensure error messages do not leak internal system state

## Style Guidelines

- Follow PEP 8 for Python code
- Use descriptive variable and function names
- Document all public functions with docstrings
- Keep functions small and focused
- Add type hints where appropriate

## License
MIT License. See [LICENSE](LICENSE) for details.