

![raggy](raggy.png)

**Single file** RAG (ChromaDB) with hybrid search, smart chunking, and normalized scoring. Enterprise-grade quality with comprehensive testing and security enhancements. 

## Quick Start

Download `raggy.py` (or the `raggy` launcher) and place it in the root of your project. All you need to do next is put all your documents for the RAG inside a `./docs` folder and **build** the RAG using the command line.

**Supported file formats:** `.md` (Markdown), `.pdf` (PDF), `.docx` (Word), `.txt` (Plain text)


```bash
# First-time setup (required)
python raggy.py init                     # Initialize environment and install dependencies

# Basic usage
python raggy.py build                    # Index your docs
python raggy.py search "your query"      # Search with normalized scores

# Or use the portable launcher (auto-manages venv & deps):
./raggy build
./raggy search "your query"

# Enhanced features  
python raggy.py search "term" --hybrid   # Hybrid semantic + keyword search
python raggy.py search "api" --expand    # Query expansion
python raggy.py search "fix" --path docs/CHANGELOG.md  # Filter by path
python raggy.py optimize                 # Benchmark search modes

# Quality assurance (new!)
python raggy.py test                     # Run built-in self-tests
python raggy.py diagnose                 # Check system health  
python raggy.py validate                 # Verify configuration
python raggy.py compact                  # Archive aged docs and refresh index
python raggy.py compact --yes            # Compact without confirmation (for scripts)

# Version management
python raggy.py --version                # Show current version
# (Automatic update notifications shown once per day when available)
```

## Configuration

1. Copy the example config:
   ```bash
   cp raggy_config_example.yaml raggy_config.yaml
   ```

2. Customize the `expansions` section with your domain terms:
   ```yaml
   search:
     expansions:
       myterm: ["myterm", "synonym1", "synonym2"]
       acronym: ["acronym", "full phrase"]
   ```

3. Use expanded search:
   ```bash
   python raggy.py search "myterm" --expand
   ```

### Update Notifications

Raggy automatically checks for updates once per day (non-intrusive):

```bash
# When an update is available, you'll see:
📦 Raggy update available: v2.1.0 → https://github.com/dimitritholen/raggy/releases/latest
```

**To disable update checks**, add to your `raggy_config.yaml`:
```yaml
updates:
  check_enabled: false
```

**Privacy**: Update checks are anonymous GitHub API calls. No tracking or personal data is sent.

> **💡 TIP**: Instead of manually creating expansions, let AI extract domain-specific terms from your documents and generate the config for you! Ask Claude or ChatGPT: 
>
> *"Analyze my documentation and create raggy_config.yaml expansions for these terms: [list key terms from your docs]"*. 
> 
> Or go **full-automatic** with:
> 
> *"Analyze my documentation in ./docs/ and its subfolders and create raggy_config.yaml expansions for the most important terms. Do not go overboard, so keep it strategic. Do it step by step. Ultrathink."* 
> 
> This saves time and ensures you capture the right synonyms and related concepts.

## Key Features

### Core RAG Functionality
- **Hybrid Search**: Combines semantic + BM25 keyword ranking for precise results
- **Smart Chunking**: Markdown-aware document processing with boundary detection
- **Normalized Scoring**: 0-1 scores with quality labels (Excellent/Good/Fair/Poor)
- **Query Expansion**: Automatic synonym expansion for domain-specific terms
- **Model Presets**: fast/balanced/multilingual/accurate options for different needs
- **Multi-Format**: Supports `.md`, `.pdf`, `.docx`, `.txt` documents

### Quality & Reliability (New!)
- **Built-in Testing**: Self-diagnostics with `python raggy.py test`
- **System Health**: Comprehensive diagnostics with `python raggy.py diagnose` and detailed status with `python raggy.py status`  
- **Security Hardened**: Path validation, input sanitization, secure hashing
- **Type Safe**: Full type hints for better IDE support and error prevention
- **Performance Optimized**: Streaming file processing and pre-compiled regex patterns
- **Comprehensive Testing**: 85%+ test coverage with unit and integration tests

### Developer Experience
- **Universal**: Drop into any project - just copy `raggy.py` (or `raggy`) 
- **💰 Completely Free**: No API costs - everything runs locally
- **🏠 100% Private**: Your documents never leave your machine
- **⚡ Fast Setup**: One command initialization with automatic dependency management
- **🔄 Auto-Updates**: Non-intrusive update notifications (once per day, easily disabled)

## Why Raggy is Fun & Cheap

**🆓 Zero Cost Forever**
- No API tokens, no monthly fees, no usage limits
- Run unlimited searches on unlimited documents
- Perfect for developers, students, and cost-conscious teams

**🏠 100% Local & Private**
- Your documents never leave your machine
- No internet required after initial setup
- No rate limits or API downtime
- Full control over your data and processing

**⚡ Blazing Fast**
- Local search = instant results
- No network latency or API delays
- Process thousands of documents without breaking the bank

**vs. Cloud RAG Services:**
- Cloud RAG: $$/year for typical usage
- Raggy: $0/year forever
- That's ∞% savings! 🎉

## Requirements

- **Python 3.8+** (tested on 3.8-3.12)
- **`uv` package manager** ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- **5-10MB disk space** for dependencies (everything installs locally)

## Setup

1. **Install uv** (if not already installed):
   ```bash
   # On macOS/Linux:
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows:
   # Download and run installer from https://astral.sh/uv/getting-started/installation/
   ```

2. **Initialize the project**:
   ```bash
   python raggy.py init
   ```
   This will:
   - Create a virtual environment (`.venv`)
   - Generate `pyproject.toml`
   - Install all dependencies
   - Create a `docs/` directory

3. **Add your documents** to the `docs/` directory  
   Supported formats: `.md`, `.pdf`, `.docx`, `.txt`

4. **Index your documents**:
   ```bash
   python raggy.py build
   # Or:
   ./raggy build
   ```

5. **Start searching**:
   ```bash
   python raggy.py search "your query"
   # Or:
   ./raggy search "your query"
   ```

## Testing & Quality Assurance

Raggy includes comprehensive testing and diagnostic tools:

### Built-in Self-Tests
```bash
python raggy.py test        # Run core functionality tests
python raggy.py diagnose    # Check system health and dependencies  
python raggy.py validate    # Verify configuration settings
```

### For Contributors & Advanced Users
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run comprehensive test suite
pytest tests/ --cov=raggy

# Run code quality checks
ruff check raggy.py         # Linting
ruff format raggy.py        # Code formatting
mypy raggy.py               # Type checking
bandit raggy.py             # Security scan
```

See [TESTING.md](TESTING.md) for detailed testing documentation.

### Continuous Integration
- ✅ **Multi-Python Testing**: Automated testing on Python 3.8-3.12
- ✅ **Security Scanning**: Vulnerability detection with Bandit and Safety
- ✅ **Code Quality**: Linting, formatting, and type checking
- ✅ **Performance Testing**: Benchmark validation and optimization checks

## What's New in v2.0

### 🔒 Security Enhancements
- **SHA256 hashing** replaces MD5 for file integrity  
- **Path validation** prevents directory traversal attacks
- **Input sanitization** with file size limits (100MB max)
- **Error message sanitization** prevents information leakage

### ⚡ Performance Improvements  
- **15-30% faster** search with pre-compiled regex patterns
- **Streaming file processing** for large documents
- **Optimized memory usage** with chunked file operations
- **Enhanced BM25 scoring** with improved tokenization
- **Portable Launcher**: New `./raggy` script for zero-config execution (auto-manages venv & dependencies)

### 🧪 Enterprise-Grade Testing
- **85%+ test coverage** with comprehensive test suite
- **Built-in diagnostics** for troubleshooting and validation  
- **Security scanning** with automated vulnerability detection
- **Multi-version compatibility** testing (Python 3.8-3.12)

### 🛠️ Developer Experience
- **Full type hints** for better IDE support and error prevention
- **Comprehensive error handling** with specific exception types
- **Improved documentation** with testing guides and examples
- **CI/CD pipeline** with automated quality checks

**Backward Compatible**: All existing raggy v1.x commands work unchanged! 

## AI Agent Integration

Raggy is designed to work seamlessly with AI coding agents (Claude, Cursor, Windsurf, etc.). To enable **Knowledge-Driven Development**, you must provide your agent with the "operating protocol" defined in `AGENTS.md`.

### Why `AGENTS.md` is Critical

`AGENTS.md` is not just a set of instructions; it is the **cortex** of your AI developer. It solves the "Cold Start" problem where agents lack context about your project's history, architecture, and active state.

By including `AGENTS.md` in your agent's system prompt or rules, you ensure:
1.  **Context Awareness**: The agent explicitly gathers context from `CURRENT_STATE.md` and RAG search *before* writing code.
2.  **Architectural Consistency**: It enforces adherence to established patterns found in your documentation.
3.  **Living Memory**: It forces the agent to update `CURRENT_STATE.md` and `CHANGELOG.md`, creating a continuous loop of knowledge preservation.

### How to Enable

1.  **Copy the content** of [AGENTS.md](AGENTS.md).
2.  **Paste it** into your agent's custom instructions file:
    - **Cursor**: `.cursorrules`
    - **Windsurf**: `~/.codeium/windsurf/memories` (or project-specific rules)
    - **Claude Projects**: Project Instructions
    - **General LLMs**: Paste it at the start of your session.

> **⚠️ IMPORTANT**: Without this protocol, your agent is just a smart code generator. WITH this protocol, it becomes a **strategic partner** that maintains your project's long-term health.
