# Autonomous & Interactive C++ Codebase Agents

A modular suite of intelligent C++ engineering agents built with **LangGraph**, **Ollama**, and **Google Gemini API**, leveraging deep AST semantic code intelligence via **`clangd-query`** and fast regex search via **`ripgrep` (`rg`)**.

---

## Agents Overview

### 1. Autonomous C++ Code Review Agent (`code_review_agent.py`)
- **Purpose**: Autonomous, exhaustive code review across 100+ file codebases without conversational context fatigue or skipped files.
- **Architecture**: Deterministic multi-node Map-Reduce orchestrator (Planning -> Module-by-Module Audit -> Final Synthesis).
- **Symbol & Class Discovery**: Scans repository headers and sources (`rg class/struct`) to index all declared classes, structs, and interfaces before exploration starts.
- **Semantic Code Inspection**: Prioritizes `clangd-query interface` (class layouts/APIs), `clangd-query show` (member implementations and algorithms), and `clangd-query usages` (cross-file reference validation), strictly avoiding full-file code dumps of large files.
- **Deterministic Python Audit Ledger & Completion Enforcement**: Eliminates premature module exit on large folders and classes with 400+ lines of interface results. Python tracks all files, declared classes, and discovered member functions. If the LLM attempts to exit prematurely without reading function implementations, Python halts the transition, reports the exact remaining functions and files, and keeps the agent in the module until all code has been audited.
- **Strict Context Management**: Real-time context token banner printing before every tool call, paired with proactive rolling summarization (`--context-summarize-threshold`) that preserves recent tool outputs while summarizing older history.
- **User Directives**: Accepts initial review directives via `--user-prompt` (`-u`), injecting targeted focus areas into module prompts and the final synthesis report.
- **Output**: Generates a structured 4-part Code Review Report (`CPP_CODE_REVIEW_REPORT.md`).

### 2. Autonomous Codebase Documentation Agent (`code_documenter_agent.py`)
- **Purpose**: Explores the repository folder-by-folder and incrementally generates publication-grade Markdown documentation.
- **Incremental Writing**: Uses `append_documentation_section` to immediately flush documented modules and classes to disk, preventing data loss and memory exhaustion.
- **Cross-Folder Architectural Memory**: Carries a structured architectural ledger across folders, connecting components and establishing data-flow relationships without redundant explanations.
- **Symbol & Class Discovery**: Discovers all classes and structs across the codebase to ensure every symbol and key function is documented at least once.
- **Targeted Semantic Exploration**: Prioritizes `clangd-query interface`, `show`, and `usages` over full-file reads.
- **Strict Context Budget**: Enforces an active context token ceiling (`--max-context-tokens`, default: 32k) and rolling history summarization (`--context-summarize-threshold`).
- **User Directives**: Accepts custom initial directives via `--user-prompt` (`-u`) to guide documentation depth, architectural perspective, or component focus.
- **Output**: Generates `CODEBASE_DOCUMENTATION.md` with Table of Contents, architecture overview, class specifications, and Mermaid diagrams.

### 3. Interactive Codebase Explainer & Tutor (`code_explainer_agent.py`)
- **Purpose**: Dedicated **interactive guide** whose sole job is to help developers deeply understand an existing codebase.
- **Behavior**: Does **not** write new code, debug, or write documentation files. Instead, it investigates how pieces fit together, answers technical questions with exact code citations (`file:line`), traces call chains, and **progressively adapts explanations** to the user's level of understanding.
- **Built-in Shortcuts**:
  - `/overview` - Summarizes architecture, components, and entry points.
  - `/explore <Symbol>` - Deep dive into a class/struct (members, inheritance, usages).
  - `/flow <Function>` - Traces end-to-end execution flow and call hierarchy.
  - `/clear` - Clears conversational context for a fresh topic.
  - `/help` - Shows command tips.
  - `/exit` or `quit` - Exits the interactive session.

### 4. Shared C++ Tools Module (`cpp_agent_tools.py`)
- Centralized reusable toolset powering all C++ agents:
  - **`clangd_query`**: Semantic AST code intelligence (`search`, `show`, `usages`, `hierarchy`, `signature`, `interface`).
  - **`ripgrep_search`**: High-performance regex text search for memory management keywords (`malloc`, `free`, `new`, `delete`, `strcpy`), concurrency primitives, and raw pointer patterns.
  - **`read_project_file`**: Bounded file reader with line-range slicing; protects context budget by capping unconstrained reads of large files (>80 lines) to 60 lines and directing models to semantic AST queries.
  - **`discover_project_classes` & `group_classes_by_module`**: Project-wide scanning and aggregation of classes/structs.
  - **`list_project_structure`**: Rapid inventory and directory mapping.
  - **`get_llm`**: Multi-model factory supporting local/offline Ollama (`num_ctx` allocation) and Google Gemini.

---

## Directory Structure

```
codereviewagent/
├── cpp_agent_tools.py         # Shared C++ tools (clangd-query, ripgrep, file reader, LLM factory)
├── code_review_agent.py       # Autonomous code review orchestrator (Map-Reduce, context summarizer)
├── code_documenter_agent.py   # Autonomous documentation agent (incremental Markdown, 32k context guard)
├── code_explainer_agent.py    # Interactive codebase explainer & tutor REPL
├── test_agent_tools.py        # Unit test suite verifying tools & agents
├── AGENT.md                   # Detailed clangd-query specifications & usage guidelines
├── README.md                  # Documentation
├── .env.example               # Environment variables template
└── sample_project/            # Test C++ project with intentional patterns
    ├── CMakeLists.txt
    ├── compile_commands.json
    ├── include/
    │   ├── order.h
    │   ├── order_repository.h
    │   ├── payment_processor.h
    │   └── session_manager.h
    └── src/
        ├── main.cpp
        ├── order_repository.cpp
        ├── payment_processor.cpp
        └── session_manager.cpp
```

---

## Installation & Setup

### 1. Prerequisites
Ensure you have the following installed on your system:
- Python 3.10+
- `clangd-query` (C++ code intelligence daemon CLI)
- `ripgrep` (`rg`)
- `cmake` & `clangd`

### 2. Python Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (if not already installed)
pip install langgraph langchain-core langchain-ollama langchain-google-genai rich python-dotenv tiktoken
```

### 3. Configure API Keys (for Gemini backend)
Copy `.env.example` to `.env` and set your API key:
```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

---

## Usage Guide

### Running the Autonomous Code Reviewer

```bash
# Basic review using Google Gemini:
python code_review_agent.py --provider gemini --model gemini-3.5-flash-lite --project-dir sample_project

# Review with custom initial prompt/focus directive:
python code_review_agent.py --provider gemini --model gemini-3.5-flash-lite \
  --project-dir /path/to/cpp_project \
  --user-prompt "Audit multithreading synchronization, race conditions in SessionManager, and exception safety in payment flows"

# Review using local Ollama with custom context ceiling and summarization trigger:
python code_review_agent.py --provider ollama --model llama3.1:8b \
  --project-dir /path/to/cpp_project \
  --max-context-tokens 32000 \
  --context-summarize-threshold 12000 \
  --user-prompt "Focus on raw pointer lifecycles and modern C++20 best practices"
```

### Running the Autonomous Codebase Documenter

```bash
# Document all core modules using Google Gemini:
python code_documenter_agent.py --provider gemini --model gemini-3.5-flash-lite --project-dir sample_project

# Document with an initial focus directive:
python code_documenter_agent.py --provider gemini --model gemini-3.5-flash-lite \
  --project-dir /path/to/cpp_project \
  --user-prompt "Emphasize concurrency guarantees, mutex locking strategies, and class inheritance hierarchies"

# Document only specific target folders while ignoring tests:
python code_documenter_agent.py --provider gemini --model gemini-3.5-flash-lite \
  --project-dir /path/to/cpp_project \
  --target-dirs "src,include" \
  --ignore-dirs "tests,benchmarks"

# Document using local Ollama with custom context limit and summarization threshold:
python code_documenter_agent.py --provider ollama --model llama3.1:8b \
  --project-dir /path/to/cpp_project \
  --max-context-tokens 32000 \
  --context-summarize-threshold 12000 \
  --target-dirs "src/core,src/engine"
```

### Running the Interactive Codebase Explainer

```bash
# Using Google Gemini API:
python code_explainer_agent.py --provider gemini --model gemini-3.5-flash-lite --project-dir sample_project

# Using Local/Offline Ollama:
python code_explainer_agent.py --provider ollama --model llama3.1:8b --project-dir /path/to/cpp_project
```

---

## CLI Options Reference

| Argument | Short Flag | Applicable Agent(s) | Description | Default |
|---|---|---|---|---|
| `--project-dir` | `-p` | All | Path to the target C++ codebase directory | `./sample_project` |
| `--provider` | | All | LLM provider backend: `gemini`, `google`, or `ollama` | `gemini` |
| `--model` | `-m` | All | Model name (e.g., `gemini-3.5-flash-lite`, `llama3.1:8b`, `qwen2.5:14b`) | Provider default |
| `--ollama-host` | | All | URL of the local Ollama server | `http://localhost:11434` |
| `--user-prompt`, `--initial-prompt` | `-u` | Reviewer, Documenter | Initial custom prompt or focus directive guiding the agent's audit or documentation | `""` (none) |
| `--max-context-tokens` | | Reviewer, Documenter | Maximum context token ceiling allocated and monitored across execution | `32000` (32k) |
| `--context-summarize-threshold` | | Reviewer, Documenter | Token threshold to proactively trigger rolling summarization of completed turns | `12000` tokens |
| `--output` | `-o` | Reviewer, Documenter | Custom file path for generated Markdown report or documentation | Default filename in project dir |
| `--target-dirs`, `--include-dirs` | | Documenter | Comma-separated list of folders to document (e.g. `src,include`). Other folders remain accessible for reference | All discovered modules |
| `--ignore-dirs` | | Reviewer, Documenter | Comma-separated directory names to ignore during planning (e.g. `tests,benchmarks,legacy`) | Standard build/vendor exclusions |
| `--max-steps` | | All | Maximum recursion execution steps per module exploration | `50` (documenter) / `500` (reviewer) / `60` (explainer) |

---

## Running Unit Tests

Run the test suite verifying `clangd-query`, `ripgrep`, shared tools, context management, and LangGraph agent graphs:
```bash
./venv/bin/python test_agent_tools.py
```
