#!/usr/bin/env python3
"""
Autonomous C++ Code Review Agent using LangGraph, clangd-query, and ripgrep.
Supports both Ollama (local/offline) and Google Gemini (API).
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from typing import Literal, Optional, List, Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# LangChain / LangGraph imports
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
console = Console()

# Global variable to store active project root for tools
_ACTIVE_PROJECT_DIR: Path = Path.cwd()


def set_active_project_dir(project_dir: Path) -> None:
    """Set the active project directory for tool executions."""
    global _ACTIVE_PROJECT_DIR
    _ACTIVE_PROJECT_DIR = project_dir.resolve()


def ensure_compile_commands(project_dir: Path) -> bool:
    """
    Ensure compile_commands.json is present in the project directory.
    If CMakeLists.txt exists but compile_commands.json is missing, run cmake to generate it.
    """
    compile_commands = project_dir / "compile_commands.json"
    cmakelists = project_dir / "CMakeLists.txt"

    if compile_commands.exists():
        return True

    if not cmakelists.exists():
        console.print(f"[yellow]Warning: No CMakeLists.txt found in {project_dir}. clangd-query may have limited functionality.[/yellow]")
        return False

    console.print("[cyan]Generating compile_commands.json via CMake...[/cyan]")
    build_dir = project_dir / "build"
    build_dir.mkdir(exist_ok=True)
    try:
        res = subprocess.run(
            ["cmake", "-B", str(build_dir), "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=60
        )
        if res.returncode == 0:
            build_cc = build_dir / "compile_commands.json"
            if build_cc.exists() and not compile_commands.exists():
                shutil.copy(str(build_cc), str(compile_commands))
            console.print("[green]compile_commands.json successfully generated.[/green]")
            return True
        else:
            console.print(f"[yellow]CMake configuration warning: {res.stderr[:200]}[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Could not automatically generate compile_commands.json: {e}[/yellow]")
    return False


# ============================================================================
# Agent Tools Definition
# ============================================================================

@tool
def clangd_query(
    command: Literal["search", "show", "usages", "hierarchy", "signature", "interface"],
    symbol_or_query: str,
    limit: Optional[int] = None
) -> str:
    """Query semantic C++ code intelligence using clangd-query CLI.
    clangd-query provides token-optimized semantic understanding of C++ code,
    including namespaces, templates, classes, functions, inheritance, and usages.

    Commands:
      - 'search': Find symbols across the project by name (single-word token, supports fuzzy matching).
                  Example: 'Order', 'SessionManager', 'process_order_payment'
      - 'show': Display full source code (both declaration from .h and definition from .cpp) of a class, struct, or method.
                Example: 'OrderRepository', 'SessionManager::create_session', 'IPaymentGateway'
      - 'usages': Find all reference/call sites of a symbol across the entire codebase.
                  Example: 'OrderRepository', 'SessionData'
      - 'hierarchy': Show type inheritance hierarchy (base classes and derived classes).
                     Example: 'StripeGateway', 'IPaymentGateway'
      - 'signature': Show function signatures with parameter types, return values, and overloads.
                     Example: 'add_order', 'process'
      - 'interface': Show only public methods and member variables of a class/struct.
                     Example: 'OrderRepository', 'PaymentProcessor'
    """
    global _ACTIVE_PROJECT_DIR

    cmd = ["clangd-query", command, symbol_or_query]
    if limit is not None and limit > 0:
        cmd.extend(["--limit", str(limit)])

    try:
        result = subprocess.run(
            cmd,
            cwd=str(_ACTIVE_PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=30
        )
        output = (result.stdout + result.stderr).strip()
        if not output:
            return f"[clangd-query {command} '{symbol_or_query}']: No output returned."
        return output
    except FileNotFoundError:
        return "Error: 'clangd-query' executable not found in PATH."
    except subprocess.TimeoutExpired:
        return f"Error: 'clangd-query {command} {symbol_or_query}' timed out after 30 seconds."
    except Exception as e:
        return f"Error executing clangd-query: {e}"


@tool
def ripgrep_search(
    pattern: str,
    path_filter: Optional[str] = None,
    case_insensitive: bool = False,
    is_regex: bool = True,
    file_names_only: bool = False,
    max_results: int = 40
) -> str:
    """Search codebase text using ripgrep (rg).
    Ideal for:
      - Locating patterns like raw pointers, malloc/free, new/delete, strcpy/sprintf, reinterpret_cast
      - Checking for synchronization primitives (mutex, lock_guard, shared_mutex, atomic)
      - Finding preprocessor directives, CMake definitions, include statements, or comments
    """
    global _ACTIVE_PROJECT_DIR

    cmd = ["rg", "--color=never", "--line-number"]
    if case_insensitive:
        cmd.append("-i")
    if not is_regex:
        cmd.append("-F")
    if file_names_only:
        cmd.append("-l")
    if max_results > 0:
        cmd.extend(["-m", str(max_results)])

    # Exclude build directories and cache
    cmd.extend(["--glob", "!build/**", "--glob", "!.cache/**"])

    cmd.append(pattern)

    if path_filter:
        cmd.append(path_filter)
    else:
        cmd.append(".")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(_ACTIVE_PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=20
        )
        output = result.stdout.strip()
        if not output:
            return f"[ripgrep '{pattern}']: No matches found."
        
        # Limit output length to prevent token overflow
        lines = output.splitlines()
        if len(lines) > max_results:
            truncated = "\n".join(lines[:max_results])
            return f"{truncated}\n... [Truncated {len(lines) - max_results} additional matches]"
        return output
    except FileNotFoundError:
        return "Error: 'rg' (ripgrep) executable not found in PATH."
    except subprocess.TimeoutExpired:
        return f"Error: ripgrep search for '{pattern}' timed out."
    except Exception as e:
        return f"Error executing ripgrep: {e}"


@tool
def read_project_file(
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None
) -> str:
    """Read contents of a file in the project (such as CMakeLists.txt, configuration files, headers, or sources).
    Line numbers are 1-indexed.
    """
    global _ACTIVE_PROJECT_DIR

    target = (_ACTIVE_PROJECT_DIR / file_path).resolve()
    # Safety check
    if not str(target).startswith(str(_ACTIVE_PROJECT_DIR)):
        return f"Error: Access denied. Cannot read outside project directory: {file_path}"

    if not target.exists():
        return f"Error: File '{file_path}' does not exist."
    if target.is_dir():
        return f"Error: '{file_path}' is a directory, not a file."

    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        start = max(1, start_line) if start_line is not None else 1
        end = min(total_lines, end_line) if end_line is not None else total_lines

        if start > total_lines:
            return f"Error: start_line {start} exceeds total lines ({total_lines})."

        selected = lines[start - 1:end]
        formatted = "".join(f"{i:4d} | {line}" for i, line in enumerate(selected, start=start))
        return f"File: {file_path} (Lines {start}-{end} of {total_lines})\n\n{formatted}"
    except Exception as e:
        return f"Error reading file '{file_path}': {e}"


# Global state trackers for incremental review and coverage
_RECORDED_FINDINGS: List[Dict[str, Any]] = []
_ALL_PROJECT_FILES: List[str] = []
_INSPECTED_FILES: set = set()


def reset_findings() -> None:
    """Reset the recorded findings store and file tracker for a new review run."""
    global _RECORDED_FINDINGS, _ALL_PROJECT_FILES, _INSPECTED_FILES
    _RECORDED_FINDINGS = []
    _ALL_PROJECT_FILES = []
    _INSPECTED_FILES = set()


@tool
def list_project_source_files() -> str:
    """Scan and return all C++ header (.h, .hpp) and source (.cpp, .cc, .cxx) files grouped by module/directory.
    Initializes the mandatory coverage checklist for the entire repository.
    """
    global _ACTIVE_PROJECT_DIR, _ALL_PROJECT_FILES
    headers = []
    sources = []
    for ext in ["*.h", "*.hpp", "*.hxx"]:
        headers.extend(sorted(_ACTIVE_PROJECT_DIR.glob(f"**/{ext}")))
    for ext in ["*.cpp", "*.cc", "*.cxx", "*.c"]:
        sources.extend(sorted(_ACTIVE_PROJECT_DIR.glob(f"**/{ext}")))

    # Filter out build and .cache directories
    headers = [str(p.relative_to(_ACTIVE_PROJECT_DIR)) for p in headers if "build" not in p.parts and ".cache" not in p.parts]
    sources = [str(p.relative_to(_ACTIVE_PROJECT_DIR)) for p in sources if "build" not in p.parts and ".cache" not in p.parts]

    _ALL_PROJECT_FILES = sorted(list(set(headers + sources)))

    # Group by parent directory
    dir_groups: Dict[str, List[str]] = {}
    for f in _ALL_PROJECT_FILES:
        parent = str(Path(f).parent)
        if parent not in dir_groups:
            dir_groups[parent] = []
        dir_groups[parent].append(f)

    output = f"Total Repository Inventory: {len(_ALL_PROJECT_FILES)} files ({len(headers)} headers, {len(sources)} sources)\n\n"
    output += "Directory / Module Breakdown:\n"
    for d, files in sorted(dir_groups.items()):
        output += f"📁 {d}/ ({len(files)} files):\n"
        for f in files[:10]:
            output += f"   - {f}\n"
        if len(files) > 10:
            output += f"   ... and {len(files) - 10} more files in {d}/\n"

    output += "\nUse 'track_review_progress' to mark files as inspected as you complete each module."
    return output


@tool
def track_review_progress(
    inspected_files: Optional[List[str]] = None
) -> str:
    """Track and report codebase review coverage.
    Pass 'inspected_files' with a list of file paths you just audited (e.g. ['src/net/tcp.cpp', 'include/net/tcp.h']).
    Returns the current audit coverage percentage and the list of remaining uninspected files/directories.
    """
    global _ALL_PROJECT_FILES, _INSPECTED_FILES

    if inspected_files:
        for f in inspected_files:
            # Match normalized relative paths
            clean_f = f.strip().lstrip("./")
            _INSPECTED_FILES.add(clean_f)

    total = len(_ALL_PROJECT_FILES)
    if total == 0:
        return "No project files registered yet. Please call 'list_project_source_files' first."

    completed = len(_INSPECTED_FILES)
    remaining = [f for f in _ALL_PROJECT_FILES if f not in _INSPECTED_FILES]
    pct = (completed / total) * 100 if total > 0 else 0

    output = f"Code Review Coverage: {completed}/{total} files audited ({pct:.1f}% complete)\n"
    if remaining:
        # Group remaining by directory
        rem_dirs: Dict[str, int] = {}
        for r in remaining:
            p = str(Path(r).parent)
            rem_dirs[p] = rem_dirs.get(p, 0) + 1
        output += "Remaining Unaudited Directories / Modules:\n"
        for d, count in sorted(rem_dirs.items()):
            output += f"  ⏳ {d}/ ({count} files remaining)\n"
        output += "\nNext files to inspect:\n" + "\n".join(f"  - {f}" for f in remaining[:8])
        if len(remaining) > 8:
            output += f"\n  ... and {len(remaining) - 8} more files."
    else:
        output += "🎉 100% COVERAGE REACHED: All repository files have been audited!"

    return output


@tool
def record_finding(
    category: Literal["architecture", "exceptional", "minor_improvement", "critical_flaw"],
    title: str,
    details: str,
    files_and_lines: str,
    recommended_fix: Optional[str] = None
) -> str:
    """Record an incremental code review finding immediately during exploration.
    Call this tool whenever you discover an architectural element, an exceptional implementation,
    a minor improvement, or a critical flaw.
    
    Args:
      category: 'architecture', 'exceptional', 'minor_improvement', or 'critical_flaw'
      title: Short descriptive headline (e.g. 'Buffer Overflow in SessionManager::create_session')
      details: Deep analysis explaining the rationale, impact, or idiom
      files_and_lines: Specific file paths and line ranges (e.g. 'src/session_manager.cpp:15-25')
      recommended_fix: Code snippet or exact guidance for resolving flaws
    """
    global _RECORDED_FINDINGS, _ACTIVE_PROJECT_DIR

    entry = {
        "category": category,
        "title": title,
        "details": details,
        "files_and_lines": files_and_lines,
        "recommended_fix": recommended_fix or "",
        "timestamp": time.time()
    }
    _RECORDED_FINDINGS.append(entry)

    # Save to incremental draft file on disk immediately
    draft_file = _ACTIVE_PROJECT_DIR / ".draft_review_findings.json"
    try:
        with open(draft_file, "w", encoding="utf-8") as f:
            json.dump(_RECORDED_FINDINGS, f, indent=2)
    except Exception:
        pass

    cat_badge = {
        "architecture": "[blue]ARCHITECTURE[/blue]",
        "exceptional": "[green]EXCEPTIONAL[/green]",
        "minor_improvement": "[yellow]MINOR IMPROVEMENT[/yellow]",
        "critical_flaw": "[red]CRITICAL FLAW[/red]"
    }.get(category, category.upper())

    console.print(f"  [bold]📝 Recorded Finding ({cat_badge}):[/bold] {title} ({files_and_lines})")
    return f"Successfully recorded finding [{category.upper()}]: '{title}'. Total findings recorded: {len(_RECORDED_FINDINGS)}"


# List of all available tools
TOOLS = [clangd_query, ripgrep_search, read_project_file, list_project_source_files, track_review_progress, record_finding]


# ============================================================================
# LLM Model Factory
# ============================================================================

def get_llm(provider: str, model_name: str, ollama_host: str = "http://localhost:11434"):
    """
    Instantiate the appropriate LLM based on provider (Ollama or Gemini).
    """
    provider = provider.lower()
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            console.print(f"[bold green]Initializing Ollama LLM:[/bold green] model='{model_name}', host='{ollama_host}'")
            return ChatOllama(
                model=model_name,
                base_url=ollama_host,
                temperature=0.1,
            )
        except ImportError:
            raise ImportError("langchain-ollama is required. Run: pip install langchain-ollama")

    elif provider in ("gemini", "google"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                console.print("[yellow]Warning: Neither GEMINI_API_KEY nor GOOGLE_API_KEY found in environment. Relying on default auth credentials if present.[/yellow]")
            console.print(f"[bold green]Initializing Google Gemini LLM:[/bold green] model='{model_name}'")
            return ChatGoogleGenerativeAI(
                model=model_name,
                api_key=api_key,
                temperature=0.1,
            )
        except ImportError:
            raise ImportError("langchain-google-genai is required. Run: pip install langchain-google-genai")
    else:
        raise ValueError(f"Unknown provider '{provider}'. Supported providers: 'ollama', 'gemini'.")


# ============================================================================
# LangGraph Workflow Construction
# ============================================================================

SYSTEM_PROMPT = """You are an expert autonomous C++ Code Review Agent specializing in modern C++ (C++17/C++20/C++23), systems programming, concurrency, memory safety, and software architecture.

════════════════════════════════════════════════════════════════════════════════
STRICT COVERAGE MANDATE (DO NOT STOP EARLY):
- You are strictly FORBIDDEN from concluding the review after inspecting only a small sample of files.
- You MUST systematically audit EVERY directory, module, and source file in the repository inventory.
- On large repositories (50–100+ files), you MUST NOT stop early. Work module-by-module across all subdirectories.
- A review that only samples 5–10% of the codebase is considered an incomplete failure.
════════════════════════════════════════════════════════════════════════════════

EXECUTION WORKFLOW:
1. **Repository Inventory & Architecture Mapping**:
   - Read `CMakeLists.txt` via `read_project_file`.
   - Call `list_project_source_files` to generate the complete checklist of all headers and sources in the project.
   - Record project architecture findings via `record_finding(category='architecture', ...)`.

2. **Systematic Module-by-Module Traversal**:
   - For every directory/module discovered in the inventory:
     a. Use `ripgrep_search` (with `path_filter` targeting the directory) to audit memory safety (`malloc`, `free`, `new`, `delete`, `strcpy`, `sprintf`, raw pointer arithmetic) and concurrency (`mutex`, `shared_mutex`, `atomic`).
     b. Use `clangd_query` (`search`, `show`, `interface`, `hierarchy`, `usages`) to explore the classes, functions, and inheritance in that module.
     c. Call `record_finding` immediately for every exceptional practice, minor improvement, or critical flaw.
     d. Call `track_review_progress(inspected_files=[...])` after completing each module to update coverage and view remaining directories.

3. **Coverage Check**:
   - Check `track_review_progress()`. If any directories or modules remain unaudited, CONTINUE exploring. Do not stop.

4. **Final Synthesis (Only When All Modules Are Audited)**:
   - Once all modules across the codebase have been reviewed, synthesize all recorded findings into the 4-part Code Review Report:
   # Comprehensive C++ Code Review Report
   ## 1. Project Architecture & Dependency Overview
   ## 2. What Is Implemented Exceptionally Well
   ## 3. What Needs Minor Improvements
   ## 4. What Is Poorly Implemented or Contains Critical Flaws (with concrete code fixes)
"""


import time
import re

def build_code_review_graph(llm, tools=TOOLS):
    """
    Build and compile the LangGraph ReAct agent graph with automatic rate limit retry.
    """
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState) -> Dict[str, Any]:
        """Agent reasoning node that calls LLM with available messages and tools."""
        messages = state["messages"]
        max_retries = 5
        base_delay = 6
        for attempt in range(1, max_retries + 1):
            try:
                response = llm_with_tools.invoke(messages)
                return {"messages": [response]}
            except Exception as e:
                err_msg = str(e)
                if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "RateLimit" in err_msg) and attempt < max_retries:
                    match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
                    wait_time = (float(match.group(1)) + 2) if match else base_delay * (2 ** (attempt - 1))
                    console.print(f"[yellow]Rate limit reached (429). Waiting {wait_time:.1f}s before retrying (attempt {attempt}/{max_retries})...[/yellow]")
                    time.sleep(wait_time)
                else:
                    raise e

    tool_node = ToolNode(tools)

    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition, ["tools", END])
    workflow.add_edge("tools", "agent")

    return workflow.compile()


def extract_text(content: Any) -> str:
    """Extract plain string content from string, list of parts, or structured objects."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    elif content is None:
        return ""
    else:
        return str(content)


def synthesize_final_report(findings: List[Dict[str, Any]], fallback_text: str, project_dir: Path) -> str:
    """
    Ensure a complete, rich Markdown report is always generated,
    combining LLM summary with structured findings from memory/disk.
    """
    if fallback_text and len(fallback_text.strip()) > 300 and "## " in fallback_text:
        return fallback_text

    # If fallback text was empty or truncated, recover from findings
    if not findings:
        draft_file = project_dir / ".draft_review_findings.json"
        if draft_file.exists():
            try:
                with open(draft_file, "r", encoding="utf-8") as f:
                    findings = json.load(f)
            except Exception:
                pass

    if findings:
        console.print("[cyan]Assembling comprehensive report from recorded findings...[/cyan]")
        arch = [f for f in findings if f.get("category") == "architecture"]
        exceptional = [f for f in findings if f.get("category") == "exceptional"]
        minor = [f for f in findings if f.get("category") == "minor_improvement"]
        critical = [f for f in findings if f.get("category") == "critical_flaw"]

        sections = ["# Comprehensive C++ Code Review Report\n"]

        # Section 1
        sections.append("## 1. Project Architecture & Dependency Overview")
        if arch:
            for item in arch:
                sections.append(f"### {item['title']}\n- **Files**: `{item['files_and_lines']}`\n\n{item['details']}\n")
        else:
            sections.append("Analysis of project build targets, include paths, and component structure.\n")

        # Section 2
        sections.append("## 2. What Is Implemented Exceptionally Well")
        if exceptional:
            for item in exceptional:
                sections.append(f"### {item['title']}\n- **Files**: `{item['files_and_lines']}`\n\n{item['details']}\n")
        else:
            sections.append("No exceptional patterns specifically recorded.\n")

        # Section 3
        sections.append("## 3. What Needs Minor Improvements")
        if minor:
            for item in minor:
                sections.append(f"### {item['title']}\n- **Files**: `{item['files_and_lines']}`\n\n{item['details']}")
                if item.get("recommended_fix"):
                    sections.append(f"\n**Recommended Improvement:**\n```cpp\n{item['recommended_fix']}\n```\n")
        else:
            sections.append("No minor improvements noted.\n")

        # Section 4
        sections.append("## 4. What Is Poorly Implemented or Contains Critical Flaws")
        if critical:
            for item in critical:
                sections.append(f"### ⚠️ {item['title']}\n- **Files**: `{item['files_and_lines']}`\n\n{item['details']}")
                if item.get("recommended_fix"):
                    sections.append(f"\n**Recommended Fix:**\n```cpp\n{item['recommended_fix']}\n```\n")
        else:
            sections.append("No critical flaws detected.\n")

        return "\n".join(sections)

    return fallback_text or "# Comprehensive C++ Code Review Report\n\nReview concluded."


# ============================================================================
# Main Execution Runner
# ============================================================================

def run_code_review(
    project_dir: str,
    provider: str = "gemini",
    model_name: Optional[str] = None,
    ollama_host: str = "http://localhost:11434",
    output_report_path: Optional[str] = None,
    max_steps: int = 45
) -> str:
    """
    Execute the autonomous C++ code review agent loop.
    """
    proj_path = Path(project_dir).resolve()
    if not proj_path.exists() or not proj_path.is_dir():
        console.print(f"[bold red]Error: Project directory '{project_dir}' does not exist.[/bold red]")
        sys.exit(1)

    set_active_project_dir(proj_path)
    ensure_compile_commands(proj_path)
    reset_findings()

    # Determine default model name if not specified
    if not model_name:
        if provider.lower() == "ollama":
            model_name = "llama3.1:8b"
        else:
            model_name = "gemini-3.5-flash-lite"

    console.print(Panel(
        f"[bold cyan]Autonomous C++ Code Review Agent[/bold cyan]\n"
        f"Project Path : [yellow]{proj_path}[/yellow]\n"
        f"Provider     : [green]{provider}[/green]\n"
        f"Model        : [green]{model_name}[/green]\n"
        f"Tools Active : [blue]clangd-query, ripgrep (rg), read_project_file, list_project_source_files, track_review_progress, record_finding[/blue]",
        title="Agent Configuration",
        border_style="cyan"
    ))

    llm = get_llm(provider=provider, model_name=model_name, ollama_host=ollama_host)
    app = build_code_review_graph(llm=llm, tools=TOOLS)

    initial_prompt = (
        f"Please begin an exhaustive autonomous C++ code review for the project located at '{proj_path}'.\n\n"
        f"MANDATORY INSTRUCTIONS:\n"
        f"1. Start by reading CMakeLists.txt and calling 'list_project_source_files' to get the full inventory of all files across all directories.\n"
        f"2. You are REQUIRED to audit EVERY module and directory in the repository. Do not sample just a few files; systematically traverse each subdirectory.\n"
        f"3. Use 'ripgrep_search' (with path_filter) and 'clangd_query' to analyze classes, headers, implementations, memory safety, and concurrency in each module.\n"
        f"4. Call 'record_finding' immediately whenever you find any architecture detail, exceptional pattern, minor flaw, or critical vulnerability.\n"
        f"5. Call 'track_review_progress(inspected_files=[...])' after auditing each directory/module to update coverage and verify remaining files.\n"
        f"6. DO NOT conclude early. Only synthesize the final 4-part Code Review Report once all directories and files in the repository have been reviewed."
    )

    initial_state: MessagesState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=initial_prompt)
        ]
    }

    console.print("\n[bold yellow]Starting Autonomous Review Workflow...[/bold yellow]\n")

    step_count = 0
    final_response_text = ""

    # Stream agent execution steps
    for step in app.stream(initial_state, {"recursion_limit": max_steps}, stream_mode="updates"):
        step_count += 1
        for node_name, node_update in step.items():
            if node_name == "agent":
                message = node_update["messages"][-1]
                text_content = extract_text(message.content)
                if text_content and len(text_content.strip()) > 100:
                    final_response_text = text_content

                if message.tool_calls:
                    for tc in message.tool_calls:
                        console.print(f"[bold magenta]▶ Tool Call ({step_count}):[/bold magenta] [cyan]{tc['name']}[/cyan]({json.dumps(tc['args'])})")
                else:
                    if text_content:
                        final_response_text = text_content
                        console.print(f"\n[bold green]✔ Agent Concluded Review ({step_count} steps)[/bold green]\n")
            elif node_name == "tools":
                for msg in node_update["messages"]:
                    raw_text = extract_text(msg.content)
                    content_preview = raw_text[:160].replace("\n", " ")
                    if len(raw_text) > 160:
                        content_preview += "..."
                    console.print(f"[dim]  ↳ Tool Result: {content_preview}[/dim]")

    # Build or enhance final report
    final_report = synthesize_final_report(_RECORDED_FINDINGS, final_response_text, proj_path)

    # Print markdown report
    console.print("\n" + "="*80)
    console.print(Panel("[bold green]Generated Code Review Report[/bold green]", border_style="green"))
    console.print(Markdown(final_report))
    console.print("="*80 + "\n")

    # Save report to file
    if output_report_path:
        out_file = Path(output_report_path).resolve()
    else:
        out_file = proj_path / "CPP_CODE_REVIEW_REPORT.md"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(final_report)

    console.print(f"[bold green]Report saved to:[/bold green] [cyan]{out_file}[/cyan]\n")
    return final_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Autonomous C++ Code Review Agent using LangGraph, clangd-query, and ripgrep."
    )
    parser.add_argument(
        "--project-dir", "-p",
        type=str,
        default="./sample_project",
        help="Path to the C++ project directory (default: ./sample_project)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        choices=["gemini", "google", "ollama"],
        help="LLM Provider: 'gemini' for Google Gemini API, 'ollama' for local Ollama (default: gemini)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model name (e.g. 'gemini-3.5-flash-lite', 'llama3.1:8b', 'qwen2.5:14b', 'qwen3.6:27b')"
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama host URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for the generated markdown review report (default: <project-dir>/CPP_CODE_REVIEW_REPORT.md)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=45,
        help="Maximum LangGraph execution recursion steps (default: 45)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_code_review(
        project_dir=args.project_dir,
        provider=args.provider,
        model_name=args.model,
        ollama_host=args.ollama_host,
        output_report_path=args.output,
        max_steps=args.max_steps
    )
