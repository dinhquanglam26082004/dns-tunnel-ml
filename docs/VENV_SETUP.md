# Virtual Environment Setup Guide

## Quick Start

### Windows
```powershell
# Create and activate
scripts\setup.bat

# Or manually activate existing environment
.venv\Scripts\activate.bat
```

### Linux / macOS
```bash
# Create and activate
bash scripts/setup.sh

# Or manually activate existing environment
source .venv/bin/activate
```

## Manual Setup

### 1. Create Virtual Environment

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Upgrade pip & Install Dependencies

```bash
# Upgrade core tools
python -m pip install --upgrade pip setuptools wheel

# Install with all optional groups
pip install -e ".[dev,serving,tracking,validation]"
```

### 3. Initialize Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files  # Test on existing files
```

## Verify Installation

### Python & pip
```bash
python --version        # Should be 3.10+
python -m pip --version
```

### Key Dependencies
```bash
python -c "import pandas, torch, fastapi; print('✓ Core deps OK')"
```

### Development Tools
```bash
ruff --version          # Linter/formatter
mypy --version          # Type checker
pre-commit --version    # Git hooks
pytest --version        # Test runner
```

## Deactivate Environment

To exit the virtual environment:
```bash
deactivate
```

## Troubleshooting

### "Command not found: python3"
- Windows: Use `python` instead of `python3`
- Linux/macOS: Install Python 3.10+ from python.org or your package manager

### "Permission denied: scripts/setup.sh"
```bash
chmod +x scripts/setup.sh
bash scripts/setup.sh
```

### Pre-commit setup fails
```bash
# Reinstall pre-commit and hooks
pip install --upgrade pre-commit
pre-commit install --install-hooks
```

### Virtual environment corrupted
```bash
# Recreate from scratch
rm -rf .venv  # (or rmdir /s .venv on Windows)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate.bat
pip install -e ".[dev,serving,tracking,validation]"
```

## Directory Structure

```
dns-tunnel-ml/
├── .venv/                  # Virtual environment (auto-created)
│   ├── bin/               # Scripts (Linux/macOS)
│   ├── Scripts/           # Scripts (Windows)
│   └── lib/               # Packages
├── src/                   # Source code
├── tests/                 # Tests
├── scripts/
│   ├── setup.sh          # Unix setup script
│   └── setup.bat         # Windows setup script
├── pyproject.toml        # Project metadata & dependencies
├── .pre-commit-config.yaml  # Pre-commit hooks config
└── ...
```

## Common Commands

```bash
# Activate environment
source .venv/bin/activate              # Linux/macOS
.venv\Scripts\activate.bat             # Windows

# Run tests
pytest tests/
pytest tests/ -v                       # Verbose
pytest tests/ --cov=src               # With coverage

# Code quality
ruff check src/                        # Lint
ruff format src/                       # Format
mypy src/                              # Type check
ruff check src/ --fix                 # Auto-fix issues

# Pre-commit
pre-commit run --all-files             # Run all hooks
pre-commit run ruff --all-files        # Run specific hook

# Install new package
pip install <package-name>
```

## CI/CD Integration

For automated testing and checks:

```bash
# Locally before pushing
pre-commit run --all-files
pytest tests/

# In CI pipeline, use:
pip install -e ".[dev]"
ruff check src/ tests/
mypy src/
pytest tests/ --cov=src
```
