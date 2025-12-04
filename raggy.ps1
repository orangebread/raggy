#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Cross-platform Raggy launcher for Windows (PowerShell).
.DESCRIPTION
    Mirrors the behavior of the Unix `./raggy` script:
      - Ensures Python 3.8+ is available
      - Creates/uses a .venv in the project root
      - Prefers uv for fast env + deps; falls back to python -m venv + pip
      - Bootstraps minimal pyproject/docs/config if missing
      - Installs/verifies core dependencies
      - Forwards all args to raggy.py, always adding --skip-deps
#>

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $RaggyArgs
)

$ErrorActionPreference = "Stop"

function Write-Info { param([string] $Message) Write-Host "[raggy] $Message" -ForegroundColor Green }
function Write-Warn { param([string] $Message) Write-Host "[raggy] $Message" -ForegroundColor Yellow }
function Write-Err  { param([string] $Message) Write-Host "[raggy] ERROR: $Message" -ForegroundColor Red }

# Move to project root (script location)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

# Platform flag (script is intended for Windows, but we compute this explicitly for clarity)
try {
    $isWindows = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
        [System.Runtime.InteropServices.OSPlatform]::Windows
    )
} catch {
    # Fallback: assume Windows when runtime detection is unavailable
    $isWindows = $true
}

# 1) Resolve Python interpreter (3.8+)
$pythonBin = $env:PYTHON_BIN
if (-not $pythonBin) {
    if (Get-Command python3 -ErrorAction SilentlyContinue) {
        $pythonBin = "python3"
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $pythonBin = "python"
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $pythonBin = "py"
    } else {
        Write-Err "Python 3.8+ not found. Please install Python and ensure it is on PATH."
        exit 1
    }
}

try {
    & $pythonBin -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Python >= 3.8 required."
        exit 1
    }
} catch {
    Write-Err "Unable to determine Python version via '$pythonBin'."
    exit 1
}

# 2) Paths
$VenvDir     = Join-Path $ScriptDir ".venv"
$VenvScripts = Join-Path $VenvDir "Scripts"
$VenvPython  = Join-Path $VenvScripts "python.exe"
$VenvPip     = Join-Path $VenvScripts "pip.exe"

# 3) Detect uv
$hasUv = $null -ne (Get-Command uv -ErrorAction SilentlyContinue)
$raggyPythonVersion = if ($env:RAGGY_PYTHON_VERSION) { $env:RAGGY_PYTHON_VERSION } else { "3.11" }

function Ensure-Venv {
    if (-not (Test-Path $VenvPython)) {
        if ($hasUv) {
            Write-Info "Creating virtual environment with uv (Python $raggyPythonVersion)..."
            & uv venv --python $raggyPythonVersion
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "uv venv failed; falling back to python -m venv"
                & $pythonBin -m venv $VenvDir
            }
        } else {
            Write-Info "Creating virtual environment with python -m venv..."
            & $pythonBin -m venv $VenvDir
        }
        if (-not (Test-Path $VenvPip)) {
            Write-Info "Bootstrapping pip in venv..."
            & $VenvPython -m ensurepip --upgrade | Out-Null
        }
    }
}

function Ensure-PyProject {
    $pyprojectPath = Join-Path $ScriptDir "pyproject.toml"
    if (-not (Test-Path $pyprojectPath)) {
        Write-Info "Creating minimal pyproject.toml..."
@"
[project]
name = "raggy-project"
version = "0.1.0"
description = "RAG project using Universal ChromaDB RAG System"
requires-python = ">=3.8"
dependencies = [
  "chromadb>=0.4.0",
  "sentence-transformers>=2.2.0",
  "PyPDF2>=3.0.0",
  "python-docx>=1.0.0",
]

[project.optional-dependencies]
magic-win = ["python-magic-bin>=0.4.14"]
magic-unix = ["python-magic"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"@ | Set-Content -Encoding UTF8 $pyprojectPath
    }
}

function Ensure-DocsAndConfig {
    $docsDir = Join-Path $ScriptDir "docs"
    if (-not (Test-Path $docsDir)) {
        New-Item -ItemType Directory -Path $docsDir | Out-Null
        Write-Info "Created docs/ directory"
    }

    $exampleConfig = Join-Path $ScriptDir "raggy_config_example.yaml"
    $realConfig    = Join-Path $ScriptDir "raggy_config.yaml"
    if (-not (Test-Path $exampleConfig) -and -not (Test-Path $realConfig)) {
        Write-Info "Creating raggy_config_example.yaml..."
@"
# raggy_config_example.yaml - Example Configuration File
# Copy this to raggy_config.yaml and customize for your domain

search:
  hybrid_weight: 0.7
  chunk_size: 1000
  chunk_overlap: 200
  rerank: true
  show_scores: true
  context_chars: 200
  max_results: 5
  expansions:
    api: ["api", "application programming interface", "rest api", "web service"]
    ml: ["ml", "machine learning", "artificial intelligence"]
    ai: ["ai", "artificial intelligence", "machine learning"]
    ui: ["ui", "user interface", "frontend", "user experience"]
    ux: ["ux", "user experience", "usability", "user interface"]

models:
  default: "all-MiniLM-L6-v2"
  fast: "paraphrase-MiniLM-L3-v2"
  multilingual: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  accurate: "all-mpnet-base-v2"

chunking:
  smart: true
  preserve_headers: true
  min_chunk_size: 300
  max_chunk_size: 1500

maintenance:
  thresholds:
    soft_chunk_limit: 8000
    hard_chunk_limit: 12000
    soft_document_limit: 300
    per_document_chunk_limit: 800
  retention:
    hot_days: 21
    min_hot_updates: 5
  paths:
    archive_dir: "archive/development_state"
    digest_dir: "summaries"
  auto_compact:
    rebuild_after_compact: true
"@ | Set-Content -Encoding UTF8 $exampleConfig
    }
}

function Ensure-DepsWithUv {
    Write-Info "Ensuring dependencies with uv pip..."
    & uv pip install `
        "chromadb>=0.4.0" `
        "sentence-transformers>=2.2.0" `
        "PyPDF2>=3.0.0" `
        "python-docx>=1.0.0" `
        "torch"
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "uv pip install failed; falling back to pip"
        Ensure-DepsWithPip
        return
    }

    if ($isWindows) {
        & uv pip install "python-magic-bin>=0.4.14" | Out-Null
    } else {
        & uv pip install "python-magic" | Out-Null
    }
}

function Ensure-DepsWithPip {
    Write-Info "Ensuring dependencies with pip (no uv detected)..."
    try {
        & $VenvPip install --upgrade pip setuptools wheel | Out-Null
    } catch {
        # Best-effort upgrade; ignore errors
    }

    & $VenvPip install `
        "chromadb>=0.4.0" `
        "sentence-transformers>=2.2.0" `
        "PyPDF2>=3.0.0" `
        "python-docx>=1.0.0" `
        "torch"

    if ($isWindows) {
        & $VenvPip install "python-magic-bin>=0.4.14" | Out-Null
    } else {
        & $VenvPip install "python-magic" | Out-Null
    }
}

function Get-PipSpecForImport {
    param([string] $Module)
    switch ($Module) {
        "chromadb"             { "chromadb>=0.4.0" }
        "sentence_transformers" { "sentence-transformers>=2.2.0" }
        "PyPDF2"               { "PyPDF2>=3.0.0" }
        "docx"                 { "python-docx>=1.0.0" }
        "torch"                { "torch" }
        "magic" {
            if ($isWindows) { "python-magic-bin>=0.4.14" } else { "python-magic" }
        }
        default                { $Module }
    }
}

function Test-PythonImport {
    param([string] $Module)
    & $VenvPython -c "import $Module" *> $null
    return ($LASTEXITCODE -eq 0)
}

function Verify-And-FixImports {
    $modules = @("chromadb", "sentence_transformers", "PyPDF2", "docx", "torch")
    $missing = @()

    foreach ($m in $modules) {
        if (-not (Test-PythonImport -Module $m)) {
            $missing += $m
        }
    }

    if ($missing.Count -gt 0) {
        foreach ($mod in $missing) {
            $spec = Get-PipSpecForImport -Module $mod
            if ($hasUv) {
                & uv pip install $spec
            } else {
                & $VenvPip install $spec
            }

            if ($mod -eq "sentence_transformers") {
                if (-not (Test-PythonImport -Module "torch")) {
                    if ($hasUv) {
                        & uv pip install "torch"
                    } else {
                        & $VenvPip install "torch"
                    }
                }
                if ($hasUv) {
                    & uv pip install "sentence-transformers>=2.2.0"
                } else {
                    & $VenvPip install "sentence-transformers>=2.2.0"
                }
            }
        }
    }

    $stillMissing = @()
    foreach ($m in $modules) {
        if (-not (Test-PythonImport -Module $m)) {
            $stillMissing += $m
        }
    }

    if ($stillMissing.Count -gt 0) {
        Write-Err "Missing required Python modules after installation: $($stillMissing -join ', ')"
        Write-Err "Please check network connectivity or package availability and re-run."
        exit 1
    }
}

function Ensure-StateFiles {
    $docsDir       = Join-Path $ScriptDir "docs"
    $currentState  = Join-Path $docsDir "CURRENT_STATE.md"
    $changelogPath = Join-Path $docsDir "CHANGELOG.md"
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    if (-not (Test-Path $currentState)) {
@"
# Current State

**Last Updated:** $ts
**RAG System:** Raggy v2.0.0

## Architecture
- **Type:** Universal ChromaDB RAG
- **Storage:** Local Vector DB (./vectordb)

## Active Features
- **Smart Chunking:** Enabled (Markdown-aware)
- **Search:** Semantic + Keyword (Hybrid)
- **Context:** Tiered (Current State + Changelog)

## Current Focus
- Initial setup and configuration

## Next Steps
1. Add documentation to docs/
2. Run \`./raggy build\`
3. Start development
"@ | Set-Content -Encoding UTF8 $currentState
        Write-Info "Created initial docs/CURRENT_STATE.md"
    }

    if (-not (Test-Path $changelogPath)) {
@"
# Project Changelog

## Update - $ts (Initialization)

COMPLETED:
- ✅ Raggy environment initialized
- ✅ Documentation structure created (CURRENT_STATE.md, CHANGELOG.md)
- ✅ Dependencies installed

DECISIONS:
- Adopted Tiered Context workflow
"@ | Set-Content -Encoding UTF8 $changelogPath
        Write-Info "Created initial docs/CHANGELOG.md"
    }
}

# Bootstrap environment
Ensure-Venv
Ensure-PyProject
Ensure-DocsAndConfig

if ($hasUv) {
    Ensure-DepsWithUv
} else {
    Ensure-DepsWithPip
}
Verify-And-FixImports

$PassthruArgs = @()
if ($RaggyArgs) { $PassthruArgs = $RaggyArgs }

# 8) Intercept `init` to guarantee bootstrap regardless of uv state
if ($PassthruArgs.Count -gt 0 -and $PassthruArgs[0] -eq "init") {
    if ($hasUv) {
        if (Test-Path $VenvDir) {
            Write-Info "Recreating virtual environment with uv (Python $raggyPythonVersion) for compatibility..."
            Remove-Item -Recurse -Force $VenvDir
        }
        & uv venv --python $raggyPythonVersion
        if ($LASTEXITCODE -ne 0) {
            Write-Err "Failed to create uv venv with Python $raggyPythonVersion."
            exit 1
        }
        # Refresh paths
        $VenvScripts = Join-Path $VenvDir "Scripts"
        $VenvPython  = Join-Path $VenvScripts "python.exe"
        $VenvPip     = Join-Path $VenvScripts "pip.exe"
    } else {
        # If system Python is 3.13+ and uv is missing, warn about possible torch wheel issues
        & $pythonBin -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)" | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Err "Python 3.13 detected and uv is not installed; PyTorch wheels may be unavailable for your platform."
            Write-Err "Install uv and re-run: https://docs.astral.sh/uv/getting-started/installation/"
            exit 1
        }
    }

    Ensure-StateFiles

    if ($hasUv) {
        Ensure-DepsWithUv
    } else {
        Ensure-DepsWithPip
    }
    Verify-And-FixImports

    Write-Info "Initialization complete. Verifying setup..."
    $raggyPyPath = Join-Path $ScriptDir "raggy.py"
    & $VenvPython $raggyPyPath "--skip-deps" "status"
    exit $LASTEXITCODE
}

# 9) Execute raggy.py from venv (regular commands)
$raggyPy = Join-Path $ScriptDir "raggy.py"
& $VenvPython $raggyPy "--skip-deps" @PassthruArgs
exit $LASTEXITCODE
