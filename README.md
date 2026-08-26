# Autonomous C++ Code Review Agent

An autonomous C++ Code Review Agent built with **LangGraph**, **Ollama**, and **Google Gemini API**, leveraging deep C++ semantic understanding via **`clangd-query`** and fast regex search via **`ripgrep` (`rg`)**.

---

## Key Features

1. **Systematic Autonomous Analysis Loop**:
   - Begins analysis by parsing `CMakeLists.txt` to discover build targets, compilation standards, include paths, and external dependencies.
   - Systematically navigates code using semantic AST understanding and fast codebase search.
2. **Specialized Tools**:
   - **`clangd-query`**: Semantic AST code intelligence (`search`, `show`, `usages`, `hierarchy`, `signature`, `interface`).
   - **`ripgrep` (`rg`)**: High-performance regex text search for memory management keywords (`malloc`, `free`, `new`, `delete`, `strcpy`), concurrency primitives, and raw pointer patterns.
   - **`read_project_file`**: Context-bounded file reader for build configurations and source files.
   - **`record_finding`**: Incremental review findings recorder that writes observations step-by-step into persistent state and `.draft_review_findings.json` as exploration happens (ideal for scaling to 50+ files without context memory limits).
3. **Multi-Model Support**:
   - **Ollama**: Run offline/locally with models such as `llama3.1:8b`, `qwen2.5:14b`, `qwen2.5:32b`, `qwen3.6:27b`.
   - **Google Gemini API**: Seamless fallback/testing mode using `gemini-2.5-flash` or `gemini-1.5-pro` when local GPU/Ollama is not available.
4. **Structured 4-Part Review Output**:
   - Project Architecture & Dependency Overview
   - What is implemented exceptionally well (RAII, Modern C++ idioms, concurrency safety)
   - What needs minor improvements (pass-by-value, magic numbers, missing virtual destructors)
   - Poor implementations & critical flaws (Use-After-Free, buffer overflows, data races, Rule of 3/5 violations)

---

## Directory Structure

```
codereviewagent/
├── code_review_agent.py      # Main autonomous LangGraph agent script & CLI
├── test_agent_tools.py       # Unit tests for clangd-query, ripgrep, and LangGraph
├── AGENT.md                  # Detailed clangd-query specifications & usage guidelines
├── README.md                 # Documentation
├── .env.example              # Environment variables template
└── sample_project/           # Test C++ project with intentional good/bad patterns
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

## Running the Agent

### A. Testing with Google Gemini API
```bash
python code_review_agent.py --provider gemini --model gemini-3.5-flash-lite --project-dir sample_project
```

### B. Running with Local Ollama
```bash
# Ensure Ollama server is running (e.g. `ollama serve`)
python code_review_agent.py --provider ollama --model llama3.1:8b --project-dir sample_project
```

You can also specify a custom Ollama host:
```bash
python code_review_agent.py --provider ollama --model qwen2.5:14b --ollama-host http://192.168.1.100:11434 --project-dir /path/to/cpp/project
```

### CLI Options

| Argument | Description | Default |
|---|---|---|
| `--project-dir`, `-p` | Path to the target C++ codebase | `./sample_project` |
| `--provider` | LLM backend: `gemini`, `google`, or `ollama` | `gemini` |
| `--model`, `-m` | Model name (e.g., `gemini-3.5-flash-lite`, `llama3.1:8b`, `qwen2.5:14b`) | `gemini-3.5-flash-lite` |
| `--ollama-host` | URL of the Ollama server | `http://localhost:11434` |
| `--ignore-dirs` | Comma-separated list of directories to ignore (e.g. `third_party,external,vendor,tests`) | Built-in third-party exclusions |
| `--output`, `-o` | Output Markdown report file path | `<project-dir>/CPP_CODE_REVIEW_REPORT.md` |
| `--max-steps` | Maximum LangGraph execution recursion steps | `500` |

---

## Running Unit Tests

Run the test suite verifying `clangd-query`, `ripgrep`, and LangGraph state graph mechanics:
```bash
./venv/bin/python3 test_agent_tools.py
```

