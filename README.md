# Autonomous & Interactive C++ Codebase Agents

A modular suite of intelligent C++ engineering agents built with **LangGraph**, **Ollama**, and **Google Gemini API**, leveraging deep AST semantic code intelligence via **`clangd-query`** and fast regex search via **`ripgrep` (`rg`)**.

---

## 🛠️ Agents Overview

### 1. Autonomous C++ Code Review Agent (`code_review_agent.py`)
- **Purpose**: Autonomous, exhaustive code review across 100+ file codebases without conversational context fatigue.
- **Architecture**: Deterministic multi-node Map-Reduce orchestrator (Planning $\rightarrow$ Module-by-Module Audit $\rightarrow$ Final Synthesis).
- **Output**: Generates a structured 4-part Code Review Report (`CPP_CODE_REVIEW_REPORT.md`).

### 2. Interactive Codebase Explainer & Tutor (`code_explainer_agent.py`)
- **Purpose**: Dedicated **interactive guide** whose sole job is to help developers deeply understand an existing codebase.
- **Behavior**: Does **not** write new code, debug, or write documentation files. Instead, it investigates how pieces fit together, answers technical questions with exact code citations (`file:line`), traces call chains, and **progressively adapts explanations** to the user's level of understanding.
- **Built-in Shortcuts**:
  - `/overview` - Summarizes architecture, components, and entry points.
  - `/explore <Symbol>` - Deep dive into a class/struct (members, inheritance, usages).
  - `/flow <Function>` - Traces end-to-end execution flow and call hierarchy.
  - `/clear` - Clears conversational context for a fresh topic.
  - `/help` - Shows command tips.
  - `/exit` or `quit` - Exits the interactive session.

### 3. Shared C++ Tools Module (`cpp_agent_tools.py`)
- Centralized reusable toolset powering any future C++ agents:
  - **`clangd_query`**: Semantic AST code intelligence (`search`, `show`, `usages`, `hierarchy`, `signature`, `interface`).
  - **`ripgrep_search`**: High-performance regex text search for memory management keywords (`malloc`, `free`, `new`, `delete`, `strcpy`), concurrency primitives, and raw pointer patterns.
  - **`read_project_file`**: Bounded file reader with line range slicing.
  - **`get_llm`**: Multi-model factory supporting local/offline Ollama and Google Gemini.

---

## Directory Structure

```
codereviewagent/
├── cpp_agent_tools.py         # Shared C++ tools (clangd-query, ripgrep, file reader, LLM factory)
├── code_review_agent.py       # Autonomous code review orchestrator (Map-Reduce)
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
pip install langgraph langchain-core langchain-ollama langchain-google-genai rich python-dotenv
```

### 3. Configure API Keys (for Gemini testing)
Copy `.env.example` to `.env` and set your API key:
```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

---

## Usage Guide

### 🚀 Running the Interactive Codebase Explainer
```bash
# Using Google Gemini API:
python code_explainer_agent.py --provider gemini --model gemini-3.5-flash-lite --project-dir sample_project

# Using Local/Offline Ollama:
python code_explainer_agent.py --provider ollama --model llama3.1:8b --project-dir /path/to/cpp_project
```

### 🔍 Running the Autonomous Code Reviewer
```bash
# Using Google Gemini API:
python code_review_agent.py --provider gemini --model gemini-3.5-flash-lite --project-dir sample_project

# Using Local/Offline Ollama:
python code_review_agent.py --provider ollama --model llama3.1:8b --project-dir /path/to/cpp_project
```

---

## CLI Options

| Argument | Description | Default |
|---|---|---|
| `--project-dir`, `-p` | Path to the target C++ codebase | `./sample_project` |
| `--provider` | LLM backend: `gemini`, `google`, or `ollama` | `gemini` |
| `--model`, `-m` | Model name (e.g., `gemini-3.5-flash-lite`, `llama3.1:8b`, `qwen2.5:14b`) | `gemini-3.5-flash-lite` |
| `--ollama-host` | URL of the Ollama server | `http://localhost:11434` |
| `--ignore-dirs` | Comma-separated directories to ignore during audit (`code_review_agent.py`) | Built-in third-party exclusions |
| `--max-steps` | Maximum recursion steps per module (`code_review_agent.py`) | `300` |

---

## Running Unit Tests

Run the test suite verifying `clangd-query`, `ripgrep`, shared tools, and LangGraph agent graphs:
```bash
./venv/bin/python3 test_agent_tools.py
```
