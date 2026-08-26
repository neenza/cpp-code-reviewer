#!/usr/bin/env python3
"""
Autonomous C++ Code Review Agent using LangGraph, clangd-query, and ripgrep.
Features a deterministic Plan-and-Execute / Map-Reduce workflow designed to
exhaustively audit repositories with 100+ files without context bloat or premature exits.
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from typing import Literal, Optional, List, Dict, Any, TypedDict, Annotated
import operator
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.markup import escape

# LangChain / LangGraph imports
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
console = Console()

# Global variable to store active project root for tools
_ACTIVE_PROJECT_DIR: Path = Path.cwd()
_RECORDED_FINDINGS: List[Dict[str, Any]] = []


def set_active_project_dir(project_dir: Path) -> None:
    """Set the active project directory for tool executions."""
    global _ACTIVE_PROJECT_DIR
    _ACTIVE_PROJECT_DIR = project_dir.resolve()


def reset_findings() -> None:
    """Reset the recorded findings store for a new review run."""
    global _RECORDED_FINDINGS
    _RECORDED_FINDINGS = []


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
      - 'show': Display full source code (both declaration and definition) of a class or function.
      - 'usages': Find all reference/call sites of a symbol across the entire codebase.
      - 'hierarchy': Show type inheritance hierarchy (base classes and derived classes).
      - 'signature': Show function signatures with parameter types, return values, and overloads.
      - 'interface': Show only public methods and member variables of a class/struct.
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
            return f"[ripgrep '{pattern}']: No matches found in {path_filter or 'project'}."
        
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

    console.print(f"  [bold]📝 Recorded Finding ({cat_badge}):[/bold] {escape(title)} ({escape(files_and_lines)})")
    return f"Successfully recorded finding [{category.upper()}]: '{title}'. Total findings recorded: {len(_RECORDED_FINDINGS)}"


MODULE_AUDIT_TOOLS = [clangd_query, ripgrep_search, read_project_file, record_finding]


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


# ============================================================================
# Deterministic Multi-Node State Graph
# ============================================================================

class RepoReviewState(TypedDict):
    project_dir: str
    all_files: List[str]
    modules: List[str]
    module_files_map: Dict[str, List[str]]
    current_module_index: int
    findings: Annotated[List[Dict[str, Any]], operator.add]
    final_report: str


DEFAULT_IGNORED_DIRS = {
    "build", ".cache", ".git", ".vscode", ".idea",
    "third_party", "thirdparty", "external", "vendor",
    "deps", "_deps", "vcpkg_installed", "conan", "submodules"
}


def is_ignored_path(rel_path: Path, custom_ignored: Optional[set] = None) -> bool:
    """Check if a path belongs to an ignored/third-party directory."""
    ignored = DEFAULT_IGNORED_DIRS.union(custom_ignored or set())
    for part in rel_path.parts:
        if part.lower() in ignored or part.startswith("."):
            return True
    return False


def discover_and_plan_node(state: RepoReviewState) -> Dict[str, Any]:
    """
    Node 1: Discover active project source files and CMake structure,
    excluding third-party/vendor libraries not managed as core codebase.
    """
    global _ACTIVE_PROJECT_DIR
    proj_path = _ACTIVE_PROJECT_DIR

    console.print("\n[bold cyan]═══ Phase 1: Repository Inventory & Module Planning ═══[/bold cyan]")

    # Check compile_commands.json for exact CMake compiled translation units
    cc_sources = set()
    for cc_file in [proj_path / "compile_commands.json", proj_path / "build" / "compile_commands.json"]:
        if cc_file.exists():
            try:
                with open(cc_file, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                for entry in entries:
                    src_file = entry.get("file")
                    if src_file:
                        p = Path(src_file)
                        if p.is_absolute():
                            try:
                                rel = p.relative_to(proj_path)
                                if not is_ignored_path(rel):
                                    cc_sources.add(str(rel))
                            except ValueError:
                                pass
                        else:
                            if not is_ignored_path(p):
                                cc_sources.add(str(p))
            except Exception:
                pass
            if cc_sources:
                break

    # Discover headers and sources in project
    headers = []
    sources = []
    for ext in ["*.h", "*.hpp", "*.hxx"]:
        headers.extend(sorted(proj_path.glob(f"**/{ext}")))
    for ext in ["*.cpp", "*.cc", "*.cxx", "*.c"]:
        sources.extend(sorted(proj_path.glob(f"**/{ext}")))

    rel_headers = []
    for p in headers:
        try:
            rel = p.relative_to(proj_path)
            if not is_ignored_path(rel):
                rel_headers.append(str(rel))
        except ValueError:
            pass

    rel_sources = []
    for p in sources:
        try:
            rel = p.relative_to(proj_path)
            if not is_ignored_path(rel):
                rel_sources.append(str(rel))
        except ValueError:
            pass

    all_files = sorted(list(set(rel_headers + rel_sources)))

    # Group files by parent directory (module)
    module_map: Dict[str, List[str]] = {}
    for f in all_files:
        parent = str(Path(f).parent)
        if parent not in module_map:
            module_map[parent] = []
        module_map[parent].append(f)

    sorted_modules = sorted(list(module_map.keys()))

    console.print(f"[green]Discovered {len(all_files)} primary project files across {len(sorted_modules)} core directories/modules:[/green]")
    if cc_sources:
        console.print(f"  [dim](Validated {len(cc_sources)} active compilation units from compile_commands.json)[/dim]")
    for mod in sorted_modules:
        console.print(f"  📁 [bold]{escape(mod)}/[/bold] ({len(module_map[mod])} files)")

    # Read CMakeLists.txt if present
    cmakelists_path = proj_path / "CMakeLists.txt"
    arch_findings = []
    if cmakelists_path.exists():
        try:
            with open(cmakelists_path, "r", encoding="utf-8") as f:
                cmake_content = f.read()
            arch_findings.append({
                "category": "architecture",
                "title": "CMake Build Configuration & Target Structure",
                "details": f"Core project structure with {len(all_files)} files across modules: {', '.join(sorted_modules)}.\nCMake configuration:\n```cmake\n{cmake_content[:500]}\n```",
                "files_and_lines": "CMakeLists.txt",
                "recommended_fix": "",
                "timestamp": time.time()
            })
            global _RECORDED_FINDINGS
            _RECORDED_FINDINGS.extend(arch_findings)
        except Exception:
            pass

    return {
        "all_files": all_files,
        "modules": sorted_modules,
        "module_files_map": module_map,
        "current_module_index": 0,
        "findings": arch_findings
    }


def build_module_reviewer(llm):
    """
    Build a focused ReAct reviewer for auditing a single directory module.
    """
    llm_with_tools = llm.bind_tools(MODULE_AUDIT_TOOLS)

    def agent_step(state: MessagesState) -> Dict[str, Any]:
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
                    import re
                    match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
                    wait_time = (float(match.group(1)) + 2) if match else base_delay * (2 ** (attempt - 1))
                    console.print(f"[yellow]Rate limit reached (429). Waiting {wait_time:.1f}s (attempt {attempt}/{max_retries})...[/yellow]")
                    time.sleep(wait_time)
                else:
                    raise e

    tool_node = ToolNode(MODULE_AUDIT_TOOLS)
    wf = StateGraph(MessagesState)
    wf.add_node("agent", agent_step)
    wf.add_node("tools", tool_node)
    wf.add_edge(START, "agent")
    wf.add_conditional_edges("agent", tools_condition, ["tools", END])
    wf.add_edge("tools", "agent")
    return wf.compile()


def review_module_node_factory(llm):
    """
    Factory creating the review_module_node.
    """
    sub_agent = build_module_reviewer(llm)

    def review_module_node(state: RepoReviewState) -> Dict[str, Any]:
        idx = state["current_module_index"]
        modules = state["modules"]
        current_module = modules[idx]
        files = state["module_files_map"].get(current_module, [])

        console.print(f"\n[bold yellow]═══ Phase 2: Auditing Module ({idx + 1}/{len(modules)}): [cyan]{escape(current_module)}/[/cyan] ({len(files)} files) ═══[/bold yellow]")

        prompt = (
            f"You are conducting a strict C++ code review for module directory: '{current_module}'\n"
            f"Files in this module:\n" + "\n".join(f"- {f}" for f in files) + "\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Use 'ripgrep_search' with path_filter='{current_module}' to audit for memory safety (malloc, free, new, delete, strcpy, sprintf, raw pointers) and concurrency (mutex, shared_mutex, atomic).\n"
            f"2. Use 'clangd_query' ('show', 'interface', 'hierarchy') to inspect the classes and functions defined in these files.\n"
            f"3. Call 'record_finding' for EVERY architectural detail, exceptional pattern, minor flaw, or critical vulnerability in this module.\n"
            f"4. Once you have audited these files and recorded findings, conclude your module review."
        )

        sub_state: MessagesState = {
            "messages": [
                SystemMessage(content="You are an expert modern C++ code auditor. Thoroughly examine the assigned module files. Call record_finding immediately for every discovery."),
                HumanMessage(content=prompt)
            ]
        }

        # Run focused mini-agent on this module (max 15 steps per module to keep context fast and fresh)
        for step in sub_agent.stream(sub_state, {"recursion_limit": 15}, stream_mode="updates"):
            for node_name, node_update in step.items():
                if node_name == "agent":
                    msg = node_update["messages"][-1]
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            console.print(f"  [magenta]▶ Tool Call:[/magenta] [cyan]{escape(tc['name'])}[/cyan]({escape(json.dumps(tc['args']))})")
                elif node_name == "tools":
                    for msg in node_update["messages"]:
                        raw_text = extract_text(msg.content)
                        preview = raw_text[:120].replace("\n", " ")
                        if len(raw_text) > 120:
                            preview += "..."
                        console.print(f"[dim]    ↳ Tool Result: {escape(preview)}[/dim]")

        pct = ((idx + 1) / len(modules)) * 100
        console.print(f"[green]✔ Finished Module '{escape(current_module)}/' ({idx + 1}/{len(modules)} modules - {pct:.1f}% complete)[/green]")

        return {
            "current_module_index": idx + 1,
            "findings": []  # Recorded findings are captured via record_finding
        }

    return review_module_node


def should_continue_modules(state: RepoReviewState) -> str:
    """Conditional router: loop to next module or move to final synthesis."""
    if state["current_module_index"] < len(state["modules"]):
        return "review_module"
    return "synthesize_report"


def synthesize_report_node(state: RepoReviewState) -> Dict[str, Any]:
    """
    Node 3: Compile and format all findings into the final comprehensive report.
    """
    global _RECORDED_FINDINGS, _ACTIVE_PROJECT_DIR
    proj_path = _ACTIVE_PROJECT_DIR

    console.print("\n[bold cyan]═══ Phase 3: Synthesizing Final Comprehensive Report ═══[/bold cyan]")

    findings = _RECORDED_FINDINGS
    if not findings:
        draft_file = proj_path / ".draft_review_findings.json"
        if draft_file.exists():
            try:
                with open(draft_file, "r", encoding="utf-8") as f:
                    findings = json.load(f)
            except Exception:
                pass

    arch = [f for f in findings if f.get("category") == "architecture"]
    exceptional = [f for f in findings if f.get("category") == "exceptional"]
    minor = [f for f in findings if f.get("category") == "minor_improvement"]
    critical = [f for f in findings if f.get("category") == "critical_flaw"]

    sections = [
        "# Comprehensive C++ Code Review Report",
        f"\n**Total Files Audited**: {len(state.get('all_files', []))} files across {len(state.get('modules', []))} directories/modules\n",
        "## 1. Project Architecture & Dependency Overview"
    ]

    if arch:
        for item in arch:
            sections.append(f"### {item['title']}\n- **Location**: `{item['files_and_lines']}`\n\n{item['details']}\n")
    else:
        sections.append("Exhaustive analysis completed across all discovered targets and directories.\n")

    sections.append("## 2. What Is Implemented Exceptionally Well")
    if exceptional:
        for item in exceptional:
            sections.append(f"### {item['title']}\n- **Location**: `{item['files_and_lines']}`\n\n{item['details']}\n")
    else:
        sections.append("No exceptional modern C++ patterns specifically noted.\n")

    sections.append("## 3. What Needs Minor Improvements")
    if minor:
        for item in minor:
            sections.append(f"### {item['title']}\n- **Location**: `{item['files_and_lines']}`\n\n{item['details']}")
            if item.get("recommended_fix"):
                sections.append(f"\n**Recommended Improvement:**\n```cpp\n{item['recommended_fix']}\n```\n")
    else:
        sections.append("No minor code quality improvements noted.\n")

    sections.append("## 4. What Is Poorly Implemented or Contains Critical Flaws")
    if critical:
        for item in critical:
            sections.append(f"### ⚠️ {item['title']}\n- **Location**: `{item['files_and_lines']}`\n\n{item['details']}")
            if item.get("recommended_fix"):
                sections.append(f"\n**Recommended Fix:**\n```cpp\n{item['recommended_fix']}\n```\n")
    else:
        sections.append("No critical flaws or memory vulnerabilities detected.\n")

    final_report = "\n".join(sections)
    return {"final_report": final_report}


def build_repo_review_orchestrator(llm):
    """
    Build the deterministic multi-node LangGraph orchestrator.
    """
    wf = StateGraph(RepoReviewState)

    wf.add_node("discover_and_plan", discover_and_plan_node)
    wf.add_node("review_module", review_module_node_factory(llm))
    wf.add_node("synthesize_report", synthesize_report_node)

    wf.add_edge(START, "discover_and_plan")
    wf.add_edge("discover_and_plan", "review_module")
    wf.add_conditional_edges("review_module", should_continue_modules, {
        "review_module": "review_module",
        "synthesize_report": "synthesize_report"
    })
    wf.add_edge("synthesize_report", END)

    return wf.compile()


# ============================================================================
# Main Execution Runner
# ============================================================================

def run_code_review(
    project_dir: str,
    provider: str = "gemini",
    model_name: Optional[str] = None,
    ollama_host: str = "http://localhost:11434",
    output_report_path: Optional[str] = None,
    max_steps: int = 500
) -> str:
    """
    Execute the deterministic multi-node autonomous C++ review orchestrator.
    """
    proj_path = Path(project_dir).resolve()
    if not proj_path.exists() or not proj_path.is_dir():
        console.print(f"[bold red]Error: Project directory '{project_dir}' does not exist.[/bold red]")
        sys.exit(1)

    set_active_project_dir(proj_path)
    ensure_compile_commands(proj_path)
    reset_findings()

    if not model_name:
        if provider.lower() == "ollama":
            model_name = "llama3.1:8b"
        else:
            model_name = "gemini-3.5-flash-lite"

    console.print(Panel(
        f"[bold cyan]Autonomous Modular C++ Code Review Orchestrator[/bold cyan]\n"
        f"Project Path : [yellow]{proj_path}[/yellow]\n"
        f"Provider     : [green]{provider}[/green]\n"
        f"Model        : [green]{model_name}[/green]\n"
        f"Architecture : [blue]Deterministic Plan & Map-Reduce Multi-Node Graph[/blue]\n"
        f"Tools Active : [blue]clangd-query, ripgrep (rg), read_project_file, record_finding[/blue]",
        title="Agent Configuration",
        border_style="cyan"
    ))

    llm = get_llm(provider=provider, model_name=model_name, ollama_host=ollama_host)
    app = build_repo_review_orchestrator(llm=llm)

    initial_state: RepoReviewState = {
        "project_dir": str(proj_path),
        "all_files": [],
        "modules": [],
        "module_files_map": {},
        "current_module_index": 0,
        "findings": [],
        "final_report": ""
    }

    result = app.invoke(initial_state, {"recursion_limit": max_steps})
    final_report = result.get("final_report", "")

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
        description="Autonomous Modular C++ Code Review Agent using LangGraph, clangd-query, and ripgrep."
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
        "--ignore-dirs",
        type=str,
        default="",
        help="Comma-separated directory names to ignore during audit (e.g. 'third_party,external,vendor,tests')"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum LangGraph execution recursion steps (default: 500)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.ignore_dirs:
        for d in args.ignore_dirs.split(","):
            if d.strip():
                DEFAULT_IGNORED_DIRS.add(d.strip().lower())

    run_code_review(
        project_dir=args.project_dir,
        provider=args.provider,
        model_name=args.model,
        ollama_host=args.ollama_host,
        output_report_path=args.output,
        max_steps=args.max_steps
    )
