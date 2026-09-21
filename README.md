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

### 4. General C++ Coding Assistant Agent (`coding_assistant_agent.py`)
- **Purpose**: A full-featured general software engineering assistant capable of investigating, creating, patching, refactoring, compiling, and testing code across a C++ project.
- **AST & Regex Intelligence**: Uses `clangd-query` (`interface`, `show`, `usages`, `hierarchy`) and `ripgrep` (`rg`) to deeply understand code before making any edits.
- **Surgical File Patching & File Creation**:
  - `write_project_file`: Creates new files or rewrites full files, automatically generating parent directories.
  - `edit_project_file`: Performs surgical search-and-replace chunk patching without rewriting whole files, validating exact indentation and whitespace.
- **Build & Verification Execution**:
  - `execute_shell_command`: Runs CMake build commands (`cmake --build build`), test suites (`ctest`), formatters, and git checks to verify edits.
- **Interactive User Permission Protocol**:
  - Protects the repository by prompting the user for interactive confirmation (`[y/N]`) before applying any file write, chunk patch, or shell command. Displays target files, diff previews, and command details. Supports `--auto-approve` (`-y`) for automated scripting.
- **Proactive Rolling Context Summarization**:
  - Dynamically monitors token consumption. When context reaches `--context-summarize-threshold` (default: 12,000 tokens), older turns are safely condensed into a structured technical context brief while keeping active tool calls and recent turns intact.
- **Dual Operating Modes**:
  - **Interactive REPL**: Conversational pair programming with rich Markdown output.
  - **Autonomous Task Mode**: Executes a single-shot prompt passed via `--prompt "..."`.

### 5. Shared C++ Tools Module (`cpp_agent_tools.py`)
- Centralized reusable toolset powering all C++ agents:
  - **`clangd_query`**: Semantic AST code intelligence (`search`, `show`, `usages`, `hierarchy`, `signature`, `interface`).
  - **`ripgrep_search`**: High-performance regex text search for memory management keywords, concurrency primitives, and raw pointer patterns.
  - **`read_project_file`**: Bounded file reader with line-range slicing; protects context budget by capping unconstrained reads of large files (>80 lines) to 60 lines and directing models to semantic AST queries.
  - **`write_project_file`**: Safe file writing with parent directory creation and interactive user permission validation.
  - **`edit_project_file`**: Exact chunk search-and-replace patching with diff display and user confirmation.
  - **`execute_shell_command`**: Project-sandboxed shell execution with timeout handling and user confirmation.
  - **`discover_project_classes` & `group_classes_by_module`**: Project-wide scanning and aggregation of classes/structs.
  - **`list_project_structure`**: Rapid inventory and directory mapping.
  - **`get_llm`**: Multi-model factory supporting local/offline Ollama (`num_ctx` allocation) and Google Gemini.

---

## Directory Structure

```
codereviewagent/
├── cpp_agent_tools.py         # Shared C++ tools (clangd-query, ripgrep, file readers/writers, patcher, shell)
├── code_review_agent.py       # Autonomous code review orchestrator (Map-Reduce, context summarizer)
├── code_documenter_agent.py   # Autonomous documentation agent (incremental Markdown, 32k context guard)
├── code_explainer_agent.py    # Interactive codebase explainer & tutor REPL
├── coding_assistant_agent.py  # General coding assistant (AST search, file patch/write, shell execution)
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

### Running the General C++ Coding Assistant

```bash
# Interactive REPL pair programming (prompts for confirmation before file writes/patches/shell commands):
python coding_assistant_agent.py sample_project --provider gemini --model gemini-2.5-flash

# Autonomous single-shot task:
python coding_assistant_agent.py sample_project --provider gemini --model gemini-2.5-flash \
  --prompt "Refactor SessionManager to eliminate strcpy and replace with bounds-checked copy, then run cmake build"

# Auto-approve modifications for headless CI or automated scripts:
python coding_assistant_agent.py /path/to/cpp_project --auto-approve \
  --prompt "Format all headers in include/ using clang-format"
```

---

## CLI Options Reference

| Argument | Short Flag | Applicable Agent(s) | Description | Default |
|---|---|---|---|---|
| `--project-dir` | `-p` | All | Path to the target C++ codebase directory | `./sample_project` |
| `--prompt` | | Assistant | Single-shot task directive. If omitted, starts interactive REPL | `""` (interactive) |
| `--provider` | | All | LLM provider backend: `gemini`, `google`, or `ollama` | `gemini` |
| `--model` | `-m` | All | Model name (e.g., `gemini-2.5-flash`, `llama3.1:8b`, `qwen2.5:14b`) | Provider default |
| `--ollama-host` | | All | URL of the local Ollama server | `http://localhost:11434` |
| `--user-prompt`, `--initial-prompt` | `-u` | Reviewer, Documenter | Initial custom prompt or focus directive guiding the agent's audit or documentation | `""` (none) |
| `--max-context-tokens` | | Reviewer, Documenter, Assistant | Maximum context token ceiling allocated and monitored across execution | `32000` (32k) |
| `--context-summarize-threshold` | | Reviewer, Documenter, Assistant | Token threshold to proactively trigger rolling summarization of completed turns | `12000` tokens |
| `--auto-approve`, `--yes` | `-y` | Assistant | Automatically grant permission for file writes, patches, and shell commands without interactive confirmation prompts | `False` |
| `--output` | `-o` | Reviewer, Documenter | Custom file path for generated Markdown report or documentation | Default filename in project dir |
| `--target-dirs`, `--include-dirs` | | Documenter | Comma-separated list of folders to document (e.g. `src,include`). Other folders remain accessible for reference | All discovered modules |
| `--ignore-dirs` | | Reviewer, Documenter | Comma-separated directory names to ignore during planning (e.g. `tests,benchmarks,legacy`) | Standard build/vendor exclusions |
| `--max-steps` | | All | Maximum recursion execution steps per module exploration | `50` (documenter) / `500` (reviewer) / `60` (explainer/assistant) |

---

## Running Unit Tests

Run the test suite verifying `clangd-query`, `ripgrep`, shared tools, context management, and LangGraph agent graphs:
```bash
./venv/bin/python test_agent_tools.py
```
