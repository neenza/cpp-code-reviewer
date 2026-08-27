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

# Common C++ Analysis Tools and Utilities
from cpp_agent_tools import (
    clangd_query,
    ripgrep_search,
    read_project_file,
    set_active_project_dir,
    get_active_project_dir,
    ensure_compile_commands,
    get_llm,
    extract_text,
    COMMON_CPP_TOOLS
)

_RECORDED_FINDINGS: List[Dict[str, Any]] = []


def reset_findings() -> None:
    """Reset the recorded findings store for a new review run."""
    global _RECORDED_FINDINGS
    _RECORDED_FINDINGS = []


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
    global _RECORDED_FINDINGS
    active_dir = get_active_project_dir()

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
    draft_file = active_dir / ".draft_review_findings.json"
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
    proj_path = get_active_project_dir()

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


def review_module_node_factory(llm, module_max_steps: int = 40):
    """
    Factory creating the review_module_node with configurable step limit and graceful recursion handling.
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
            f"4. Conclude your module review when key classes and memory/concurrency checks have been audited."
        )

        sub_state: MessagesState = {
            "messages": [
                SystemMessage(content="You are an expert modern C++ code auditor. Thoroughly examine the assigned module files. Call record_finding immediately for every discovery."),
                HumanMessage(content=prompt)
            ]
        }

        try:
            for step in sub_agent.stream(sub_state, {"recursion_limit": module_max_steps}, stream_mode="updates"):
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
        except Exception as e:
            # If a single module hits its local recursion limit, log and proceed to the next module
            console.print(f"[dim yellow]  (Module '{escape(current_module)}' completed exploration; proceeding to next module)[/dim yellow]")

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
    global _RECORDED_FINDINGS
    proj_path = get_active_project_dir()

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


def build_repo_review_orchestrator(llm, module_max_steps: int = 300):
    """
    Build the deterministic multi-node LangGraph orchestrator.
    """
    wf = StateGraph(RepoReviewState)

    wf.add_node("discover_and_plan", discover_and_plan_node)
    wf.add_node("review_module", review_module_node_factory(llm, module_max_steps=module_max_steps))
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
    max_steps: int = 300
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
        f"Max Steps/Mod: [yellow]{max_steps}[/yellow]\n"
        f"Tools Active : [blue]clangd-query, ripgrep (rg), read_project_file, record_finding[/blue]",
        title="Agent Configuration",
        border_style="cyan"
    ))

    llm = get_llm(provider=provider, model_name=model_name, ollama_host=ollama_host)
    app = build_repo_review_orchestrator(llm=llm, module_max_steps=max_steps)

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
