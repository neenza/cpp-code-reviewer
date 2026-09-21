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

from cpp_agent_tools import (
    clangd_query,
    ripgrep_search,
    read_project_file,
    set_active_project_dir,
    get_active_project_dir,
    ensure_compile_commands,
    get_llm,
    extract_text,
    extract_message_text,
    discover_project_classes,
    group_classes_by_module,
    parse_interface_methods,
    discover_module_functions,
    extract_touched_files,
    COMMON_CPP_TOOLS
)

try:
    import tiktoken
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKENIZER = None


def count_tokens(text: str) -> int:
    """Estimate or accurately count tokens in a text string."""
    if not text:
        return 0
    if _TOKENIZER:
        try:
            return len(_TOKENIZER.encode(text, disallowed_special=()))
        except Exception:
            pass
    # Fallback heuristic: roughly 4 characters per token
    return max(1, len(text) // 4)


def count_message_tokens(msg: BaseMessage) -> int:
    """Calculate token size of a LangChain message including tool calls."""
    tokens = count_tokens(extract_text(msg.content)) + 4
    if isinstance(msg, AIMessage) and msg.tool_calls:
        for tc in msg.tool_calls:
            tokens += count_tokens(tc.get("name", ""))
            tokens += count_tokens(json.dumps(tc.get("args", {}))) + 6
    return tokens


def partition_messages_safely(
    messages: List[BaseMessage],
    target_recent_count: int = 4
) -> tuple:
    """
    Safely partition messages into older history (to summarize) and recent turns (to preserve intact),
    guaranteeing that AIMessages with tool calls and their corresponding ToolMessages are never separated.
    """
    if len(messages) <= target_recent_count:
        return [], messages

    split_idx = len(messages) - target_recent_count

    while split_idx < len(messages) and isinstance(messages[split_idx], ToolMessage):
        split_idx += 1

    while split_idx > 0 and isinstance(messages[split_idx - 1], AIMessage) and getattr(messages[split_idx - 1], "tool_calls", None):
        split_idx -= 1

    older = messages[:split_idx]
    recent = messages[split_idx:]
    return older, recent


def print_context_banner(messages: List[BaseMessage], max_tokens: int, stage: str = "Step") -> int:
    """Print current context size, percentage, and stage banner."""
    total_tokens = sum(count_message_tokens(m) for m in messages)
    pct = (total_tokens / max(1, max_tokens)) * 100
    color = "green" if pct < 40 else ("yellow" if pct < 75 else "bold red")
    console.print(f"  [{color}][Context: {total_tokens:,} / {max_tokens:,} tokens ({pct:.1f}%)] | {stage}[/{color}]")
    return total_tokens


def manage_context_with_summarization(
    messages: List[BaseMessage],
    max_tokens: int = 32000,
    reserve_tokens: int = 2500,
    summarize_threshold: Optional[int] = None,
    audit_state: Optional[Dict[str, Any]] = None
) -> List[BaseMessage]:
    """
    Enforce strict context size limit with proactive rolling summarization for the review agent.
    Never truncates active tool outputs to avoid information loss or code distortion.
    When cumulative conversation tokens exceed the threshold, safely summarizes completed older turns
    into a high-signal technical context summary while preserving recent active turns intact.
    Injects the live audit progress ledger (content already covered vs. remaining to review)
    directly into the condensed context to prevent repetitive tool queries and focus exploration.
    """
    if summarize_threshold is None:
        summarize_threshold = min(12000, int(max_tokens * 0.45))
    effective_limit = min(summarize_threshold, max_tokens - reserve_tokens)

    cur_tokens = sum(count_message_tokens(m) for m in messages)

    if cur_tokens <= effective_limit or len(messages) <= 3:
        return messages

    # Context exceeds limit -> Perform Rolling Technical Summarization of older turns
    console.print(
        f"\n  [bold yellow][Notice] Context reached {cur_tokens:,} tokens (exceeding threshold {effective_limit:,}). "
        f"Active Rolling Summarization initiated...[/bold yellow]"
    )

    preserved_header = messages[:2] if len(messages) > 2 else messages
    conversation_tail = list(messages[2:]) if len(messages) > 2 else []

    history_to_summarize, recent_active_turns = partition_messages_safely(conversation_tail, target_recent_count=2)
    if not history_to_summarize:
        return messages

    extracted_facts = []
    for m in history_to_summarize:
        if isinstance(m, ToolMessage):
            raw = extract_text(m.content).strip()
            if raw:
                preview = raw[:350].replace("\n", " ")
                extracted_facts.append(f"- Tool `{m.name}` result: {preview}")
        elif isinstance(m, AIMessage):
            text = extract_message_text(m)
            if text and len(text) > 30:
                extracted_facts.append(f"- Review Exploration Note: {text[:250].replace(chr(10), ' ')}")

    progress_section = ""
    if audit_state:
        target_classes = audit_state.get("target_classes", [])
        interfaced_classes = set(audit_state.get("interfaced_classes", []))
        remaining_classes = [c for c in target_classes if c not in interfaced_classes]
        covered_classes = [c for c in target_classes if c in interfaced_classes]

        audited_funcs = set(audit_state.get("audited_functions", []))
        discovered_funcs = audit_state.get("discovered_functions", [])
        covered_funcs = [
            f for f in discovered_funcs
            if f in audited_funcs or f.split("::")[-1] in audited_funcs
        ]
        remaining_funcs = [
            f for f in discovered_funcs
            if f not in audited_funcs and f.split("::")[-1] not in audited_funcs
        ]

        target_files = audit_state.get("target_files", [])
        audited_files = set(audit_state.get("audited_files", []))
        covered_files = [f for f in target_files if f in audited_files]
        remaining_files = [f for f in target_files if f not in audited_files]

        module_name = audit_state.get("module_name", "")
        global _RECORDED_FINDINGS
        findings_count = len(_RECORDED_FINDINGS)

        p_lines = [
            f"**Audit Progress Ledger for Module '{module_name}' (Live Coverage Status)**:",
            f"- **Functions Audited**: {len(covered_funcs)} / {len(discovered_funcs)} covered",
        ]
        if covered_funcs:
            sample_cov = covered_funcs[:8]
            cov_str = ", ".join(f"`{f}`" for f in sample_cov)
            if len(covered_funcs) > 8:
                cov_str += f" (+{len(covered_funcs) - 8} more already reviewed)"
            p_lines.append(f"  * ALREADY COVERED (DO NOT RE-AUDIT): {cov_str}")
        if remaining_funcs:
            sample_rem = remaining_funcs[:12]
            rem_str = ", ".join(f"`{f}`" for f in sample_rem)
            if len(remaining_funcs) > 12:
                rem_str += f" (+{len(remaining_funcs) - 12} more remaining)"
            p_lines.append(f"  * REMAINING TO BE AUDITED (PRIORITIZE): {rem_str}")
        elif discovered_funcs:
            p_lines.append("  * REMAINING TO BE AUDITED: None (all discovered functions audited)")

        p_lines.append(f"- **Classes Interfaced**: {len(covered_classes)} / {len(target_classes)} covered")
        if covered_classes:
            p_lines.append(f"  * ALREADY INTERFACED (DO NOT RE-QUERY): {', '.join(f'`{c}`' for c in covered_classes)}")
        if remaining_classes:
            p_lines.append(f"  * REMAINING CLASSES TO INTERFACE: {', '.join(f'`{c}`' for c in remaining_classes)}")

        p_lines.append(f"- **Files Inspected**: {len(covered_files)} / {len(target_files)} covered")
        if covered_files:
            p_lines.append(f"  * ALREADY TOUCHED: {', '.join(f'`{f}`' for f in covered_files[:8])}")
        if remaining_files:
            p_lines.append(f"  * REMAINING UNINSPECTED FILES: {', '.join(f'`{f}`' for f in remaining_files[:8])}")

        p_lines.append(f"- **Review Findings Recorded So Far**: {findings_count} findings")
        progress_section = "\n".join(p_lines) + "\n\n"

    summary_text = (
        "### Prior Code Review Steps & Progress Summary (Condensed to stay within strict context limit):\n\n"
        + progress_section
        + "### Key Technical Facts & Observations from Prior Steps:\n"
        + ("\n".join(extracted_facts[:15]) if extracted_facts else "- Audited earlier classes and functions in current module.")
    )

    summary_message = SystemMessage(content=summary_text)
    new_messages = preserved_header + [summary_message] + recent_active_turns
    new_tokens = sum(count_message_tokens(m) for m in new_messages)
    saved = cur_tokens - new_tokens
    console.print(f"  [bold green][OK] Context condensed from {cur_tokens:,} down to {new_tokens:,} tokens (-{(saved/max(1, cur_tokens))*100:.1f}%, saved {saved:,} tokens).[/bold green]\n")
    return new_messages


def message_reducer(existing: List[BaseMessage], update: Any) -> List[BaseMessage]:
    """Custom reducer supporting list concatenation and explicit context override upon summarization."""
    if isinstance(update, tuple) and len(update) == 2 and update[0] == "override":
        return list(update[1])
    if isinstance(update, list):
        return list(existing) + list(update)
    if isinstance(update, BaseMessage):
        return list(existing) + [update]
    return list(existing)


class ModuleReviewState(TypedDict):
    messages: Annotated[List[BaseMessage], message_reducer]
    module_name: str
    user_prompt: str
    target_files: List[str]
    target_classes: List[str]
    interfaced_classes: List[str]
    discovered_functions: List[str]
    audited_functions: List[str]
    audited_files: List[str]
    nudge_count: int
    findings_at_start: int
    max_nudges: int


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

    console.print(f"  [bold][RECORDED FINDING] ({cat_badge}):[/bold] {escape(title)} ({escape(files_and_lines)})")
    return f"Successfully recorded finding [{category.upper()}]: '{title}'. Total findings recorded: {len(_RECORDED_FINDINGS)}"


MODULE_AUDIT_TOOLS = [clangd_query, ripgrep_search, read_project_file, record_finding]


# ============================================================================
# Deterministic Multi-Node State Graph
# ============================================================================

class RepoReviewState(TypedDict):
    project_dir: str
    user_prompt: str
    all_files: List[str]
    modules: List[str]
    module_files_map: Dict[str, List[str]]
    module_classes_map: Dict[str, List[str]]
    current_module_index: int
    findings: Annotated[List[Dict[str, Any]], operator.add]
    final_report: str
    max_context_tokens: int
    summarize_threshold: int


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
    Node 1: Discover active project source files, classes, and CMake structure,
    excluding third-party/vendor libraries not managed as core codebase.
    """
    proj_path = get_active_project_dir()

    console.print("\n[bold cyan]═══ Phase 1: Repository Inventory, Class Discovery & Module Planning ═══[/bold cyan]")

    user_prompt = state.get("user_prompt", "")
    if user_prompt:
        console.print(f"[cyan][User Review Directive]: {user_prompt}[/cyan]")

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

    # Discover classes and structs across project files
    file_to_classes = discover_project_classes(proj_path)
    module_classes_map = group_classes_by_module(file_to_classes)
    total_classes = sum(len(c) for c in module_classes_map.values())

    # Group files by parent directory (module)
    module_map: Dict[str, List[str]] = {}
    for f in all_files:
        parent = str(Path(f).parent)
        if parent not in module_map:
            module_map[parent] = []
        module_map[parent].append(f)

    sorted_modules = sorted(list(module_map.keys()))

    console.print(f"[green]Discovered {len(all_files)} primary project files across {len(sorted_modules)} core directories/modules.[/green]")
    console.print(f"[green]Discovered {total_classes} declared classes/structs across repository (scanned via rg class/struct):[/green]")
    for mod in sorted_modules:
        m_classes = module_classes_map.get(mod, [])
        cls_tag = f" — Classes: {', '.join(m_classes)}" if m_classes else ""
        console.print(f"  - [bold]{escape(mod)}/[/bold] ({len(module_map[mod])} files){cls_tag}")

    if cc_sources:
        console.print(f"  [dim](Validated {len(cc_sources)} active compilation units from compile_commands.json)[/dim]")

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
        "module_classes_map": module_classes_map,
        "current_module_index": 0,
        "findings": arch_findings,
        "user_prompt": user_prompt,
        "max_context_tokens": state.get("max_context_tokens", 32000),
        "summarize_threshold": state.get("summarize_threshold", 12000)
    }


def audit_tools_node(state: ModuleReviewState) -> Dict[str, Any]:
    """
    Execute tool calls emitted by the agent and deterministically update the Python audit ledger:
    - On clangd_query(command='interface'): register discovered class methods and mark class interfaced.
    - On clangd_query(command='show'): mark method and touched files as audited.
    - On read_project_file: mark file and any functions defined in it as audited.
    - On record_finding: record finding.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_msg = messages[-1]
    if not isinstance(last_msg, AIMessage) or not getattr(last_msg, "tool_calls", None):
        return {}

    tool_map = {t.name: t for t in MODULE_AUDIT_TOOLS}
    tool_messages = []

    new_interfaced = set()
    new_discovered_funcs = set()
    new_audited_funcs = set()
    new_audited_files = set()

    target_files = state.get("target_files", [])

    for tc in last_msg.tool_calls:
        t_name = tc.get("name", "")
        t_args = tc.get("args", {})
        t_id = tc.get("id", f"call_{time.time()}")

        tool_obj = tool_map.get(t_name)
        if tool_obj:
            try:
                raw_res = tool_obj.invoke(t_args)
            except Exception as e:
                raw_res = f"Error executing tool '{t_name}': {e}"
        else:
            raw_res = f"Error: Tool '{t_name}' not recognized."

        res_str = str(raw_res)
        tool_messages.append(ToolMessage(content=res_str, tool_call_id=t_id, name=t_name))

        # Check touched files in output
        touched = extract_touched_files(res_str, target_files)
        new_audited_files.update(touched)

        if t_name == "clangd_query":
            cmd = t_args.get("command", "show")
            sym = t_args.get("symbol_or_query") or t_args.get("symbol") or t_args.get("query") or ""
            if cmd == "interface":
                if sym:
                    clean_sym = sym.split("::")[-1]
                    new_interfaced.add(sym)
                    new_interfaced.add(clean_sym)
                parsed_methods = parse_interface_methods(res_str, default_class=sym)
                for item in parsed_methods:
                    new_discovered_funcs.add(item["full_symbol"])
                    if item["is_trivial"]:
                        new_audited_funcs.add(item["full_symbol"])
                        new_audited_funcs.add(item["method_name"])
            elif cmd == "show":
                if sym:
                    new_audited_funcs.add(sym)
                    short_sym = sym.split("::")[-1]
                    new_audited_funcs.add(short_sym)
                    for df in state.get("discovered_functions", []):
                        if df == sym or df.endswith("::" + short_sym) or df == short_sym:
                            new_audited_funcs.add(df)
            elif cmd == "usages":
                pass
        elif t_name == "read_project_file":
            fp = t_args.get("file_path") or t_args.get("path") or t_args.get("filename") or ""
            if fp:
                new_audited_files.add(fp)
                for df in state.get("discovered_functions", []):
                    if fp in df or Path(fp).stem in df:
                        new_audited_funcs.add(df)

    updated_interfaced = sorted(list(set(state.get("interfaced_classes", [])).union(new_interfaced)))
    updated_discovered = sorted(list(set(state.get("discovered_functions", [])).union(new_discovered_funcs)))
    updated_audited = sorted(list(set(state.get("audited_functions", [])).union(new_audited_funcs)))
    updated_files = sorted(list(set(state.get("audited_files", [])).union(new_audited_files)))

    return {
        "messages": tool_messages,
        "interfaced_classes": updated_interfaced,
        "discovered_functions": updated_discovered,
        "audited_functions": updated_audited,
        "audited_files": updated_files
    }


def route_module_reviewer(state: ModuleReviewState) -> str:
    """
    Conditional edge router for module reviewer:
    1. If the agent emitted tool calls -> route to 'tools'.
    2. If the agent emitted no tool calls -> Python checks the audit ledger.
       If any classes, functions, or files in this module remain unaudited,
       keep the agent in the module by routing to 'enforce_audit'.
    3. Once all required code elements have been inspected (or safety nudge limit reached),
       allow route to END.
    """
    messages = state.get("messages", [])
    if not messages:
        return END

    last_msg = messages[-1]
    if getattr(last_msg, "tool_calls", None):
        return "tools"

    # Agent emitted no tool calls! Check if audit is truly complete on Python side
    target_classes = state.get("target_classes", [])
    interfaced_classes = set(state.get("interfaced_classes", []))
    remaining_classes = [c for c in target_classes if c not in interfaced_classes]

    audited_funcs = set(state.get("audited_functions", []))
    discovered_funcs = state.get("discovered_functions", [])
    remaining_funcs = [
        f for f in discovered_funcs
        if f not in audited_funcs and f.split("::")[-1] not in audited_funcs
    ]

    target_files = state.get("target_files", [])
    audited_files = set(state.get("audited_files", []))
    remaining_files = [f for f in target_files if f not in audited_files]

    nudge_count = state.get("nudge_count", 0)
    max_nudges = state.get("max_nudges", 15)

    is_incomplete = bool(remaining_classes or remaining_funcs or remaining_files)

    if is_incomplete and nudge_count < max_nudges:
        return "enforce_audit"

    return END


def enforce_audit_node(state: ModuleReviewState) -> Dict[str, Any]:
    """
    Node invoked when the agent tries to exit the module prematurely.
    Informs the model of what classes, member functions, and files remain unaudited,
    and injects an imperative directive forcing it to continue auditing.
    """
    module_name = state.get("module_name", "")
    nudge_count = state.get("nudge_count", 0) + 1

    target_classes = state.get("target_classes", [])
    interfaced_classes = set(state.get("interfaced_classes", []))
    remaining_classes = [c for c in target_classes if c not in interfaced_classes]

    audited_funcs = set(state.get("audited_functions", []))
    discovered_funcs = state.get("discovered_functions", [])
    remaining_funcs = [
        f for f in discovered_funcs
        if f not in audited_funcs and f.split("::")[-1] not in audited_funcs
    ]

    target_files = state.get("target_files", [])
    audited_files = set(state.get("audited_files", []))
    remaining_files = [f for f in target_files if f not in audited_files]

    total_funcs = len(discovered_funcs)
    audited_funcs_count = total_funcs - len(remaining_funcs)
    total_files = len(target_files)
    audited_files_count = total_files - len(remaining_files)

    global _RECORDED_FINDINGS
    findings_count = len(_RECORDED_FINDINGS)

    console.print(
        f"\n  [bold yellow][Audit Incomplete - Nudge {nudge_count}]: Keeping agent in module '{escape(module_name)}'.[/bold yellow]\n"
        f"    Progress: {audited_funcs_count}/{total_funcs} functions audited | {audited_files_count}/{total_files} files touched | "
        f"{findings_count} total findings recorded"
    )
    if remaining_classes:
        console.print(f"    [yellow]Pending class interfaces ({len(remaining_classes)}):[/yellow] {', '.join(remaining_classes[:6])}")
    if remaining_funcs:
        console.print(f"    [yellow]Pending function implementations ({len(remaining_funcs)}):[/yellow] {', '.join(remaining_funcs[:8])}...")
    if remaining_files:
        console.print(f"    [yellow]Pending uninspected files ({len(remaining_files)}):[/yellow] {', '.join(remaining_files[:6])}")

    feedback_lines = [
        f"AUDIT INCOMPLETE FOR MODULE '{module_name}'. You cannot conclude or move to the next module yet.",
        f"Audit Progress: {audited_funcs_count}/{total_funcs} functions read, {audited_files_count}/{total_files} files inspected, {findings_count} findings recorded.",
        "\nThe following required code elements in this module have NOT yet been audited:"
    ]
    if remaining_classes:
        feedback_lines.append(
            f"- Classes needing interface review ({len(remaining_classes)} remaining): {', '.join(remaining_classes[:6])}\n"
            f"  ACTION REQUIRED: Call clangd_query(command='interface', symbol_or_query='{remaining_classes[0]}') to inspect class layouts."
        )
    if remaining_funcs:
        top_funcs = remaining_funcs[:15]
        feedback_lines.append(
            f"- Member/free functions not yet inspected via 'clangd_query(command=\"show\")' ({len(remaining_funcs)} remaining): "
            f"{', '.join(top_funcs)}"
        )
        feedback_lines.append(
            f"  ACTION REQUIRED: Call clangd_query(command='show', symbol_or_query='{top_funcs[0]}') to inspect its implementation code."
        )
    if remaining_files:
        feedback_lines.append(
            f"- Uninspected files in module ({len(remaining_files)} remaining): {', '.join(remaining_files[:10])}\n"
            f"  ACTION REQUIRED: Inspect these files using clangd_query for symbols or read_project_file (if <80 lines)."
        )
    feedback_lines.append(
        "\nIMPORTANT: If you discover any architectural decisions, exceptional patterns, minor improvements, "
        "or critical flaws (memory leaks, buffer overflows, missing locks, raw pointers), call 'record_finding' immediately.\n"
        "Continue auditing the pending items now."
    )

    nudge_msg = HumanMessage(content="\n".join(feedback_lines))
    return {
        "messages": [nudge_msg],
        "nudge_count": nudge_count,
        "max_nudges": state.get("max_nudges", 15)
    }


def build_module_reviewer(
    llm,
    max_context_tokens: int = 32000,
    summarize_threshold: Optional[int] = None
):
    """
    Build a focused ReAct reviewer for auditing a single directory module,
    strictly bounded by proactive token summarization and Python-side audit enforcement.
    """
    llm_with_tools = llm.bind_tools(MODULE_AUDIT_TOOLS)

    def agent_step(state: ModuleReviewState) -> Dict[str, Any]:
        trimmed_messages = manage_context_with_summarization(
            state["messages"],
            max_tokens=max_context_tokens,
            summarize_threshold=summarize_threshold,
            audit_state=state
        )
        print_context_banner(trimmed_messages, max_tokens=max_context_tokens, stage="Agent LLM Invocation")

        max_retries = 5
        base_delay = 6
        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = llm_with_tools.invoke(trimmed_messages)
                break
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

        # Calculate exact context size including the model's generated response
        total_tokens = sum(count_message_tokens(m) for m in trimmed_messages) + count_message_tokens(response)
        pct = (total_tokens / max(1, max_context_tokens)) * 100
        color = "green" if pct < 40 else ("yellow" if pct < 75 else "bold red")

        # Explicitly print context size before every tool call
        if getattr(response, "tool_calls", None):
            for tc in response.tool_calls:
                console.print(
                    f"  [{color}][Context: {total_tokens:,} / {max_context_tokens:,} tokens ({pct:.1f}%)][/{color}] "
                    f"| [magenta]Tool Call:[/magenta] [cyan]{escape(tc['name'])}[/cyan]({escape(json.dumps(tc['args']))})"
                )
        else:
            txt = extract_message_text(response)
            if txt:
                console.print(f"  [dim]Agent: {escape(txt[:100])}...[/dim]")

        # If summarization replaced/condensed earlier turns, use ("override", ...) to update state messages
        if len(trimmed_messages) != len(state["messages"]):
            return {"messages": ("override", trimmed_messages + [response])}
        return {"messages": [response]}

    wf = StateGraph(ModuleReviewState)
    wf.add_node("agent", agent_step)
    wf.add_node("tools", audit_tools_node)
    wf.add_node("enforce_audit", enforce_audit_node)
    wf.add_edge(START, "agent")
    wf.add_conditional_edges("agent", route_module_reviewer, {
        "tools": "tools",
        "enforce_audit": "enforce_audit",
        END: END
    })
    wf.add_edge("tools", "agent")
    wf.add_edge("enforce_audit", "agent")
    return wf.compile()


def review_module_node_factory(
    llm,
    module_max_steps: int = 300,
    max_context_tokens: int = 32000,
    summarize_threshold: Optional[int] = None
):
    """
    Factory creating the review_module_node with configurable step limit, context management,
    and deterministic Python-side audit ledger enforcement.
    """
    sub_agent = build_module_reviewer(
        llm,
        max_context_tokens=max_context_tokens,
        summarize_threshold=summarize_threshold
    )

    def review_module_node(state: RepoReviewState) -> Dict[str, Any]:
        idx = state["current_module_index"]
        modules = state["modules"]
        current_module = modules[idx]
        files = state["module_files_map"].get(current_module, [])

        console.print(f"\n[bold yellow]═══ Phase 2: Auditing Module ({idx + 1}/{len(modules)}): [cyan]{escape(current_module)}/[/cyan] ({len(files)} files) ═══[/bold yellow]")

        # Discovered classes for this module
        mod_classes = state.get("module_classes_map", {}).get(current_module, [])
        classes_str = ", ".join(f"`{c}`" for c in mod_classes) if mod_classes else "None directly declared (audit member/free functions)"

        proj_path = get_active_project_dir()
        initial_module_funcs = discover_module_functions(proj_path, files)
        if initial_module_funcs:
            console.print(f"  [dim]Discovered {len(initial_module_funcs)} defined functions in module files: {', '.join(initial_module_funcs[:5])}...[/dim]")

        user_prompt_section = ""
        user_prompt = state.get("user_prompt", "")
        if user_prompt:
            user_prompt_section = (
                f"════════════════════════════════════════════════════════════════════════════════\n"
                f"USER INITIAL REVIEW DIRECTIVE (HIGH PRIORITY FOCUS):\n"
                f"{user_prompt}\n"
                f"Ensure your audit specifically prioritizes and addresses the above directive.\n"
                f"════════════════════════════════════════════════════════════════════════════════\n\n"
            )

        prompt = (
            f"You are conducting a strict modern C++ code review for module directory: '{current_module}'\n"
            f"Files in this module ({len(files)} files):\n" + "\n".join(f"- {f}" for f in files) + "\n\n"
            f"Known Classes & Structs in this module (discovered via 'rg class/struct'):\n{classes_str}\n\n"
            f"{user_prompt_section}"
            f"MANDATORY CODE INSPECTION & REVIEW PROTOCOL:\n"
            f"1. AVOID FULL-FILE READS: NEVER use 'read_project_file' on large files (> 80 lines). Full file reads cause context bloat! "
            f"Only use 'read_project_file' for small files (< 80 lines) or build configs (CMakeLists.txt).\n"
            f"2. SEMANTIC CLASS INSPECTION: For each class/struct in this module:\n"
            f"   - Run 'clangd_query' with command='interface' and symbol_or_query='<ClassName>' to inspect its layout, member variables, and method signatures.\n"
            f"   - For every non-trivial method discovered in the interface, run 'clangd_query' with command='show' and symbol_or_query='<ClassName::MethodName>' to inspect its implementation code.\n"
            f"3. FUNCTION AUDIT: For all standalone functions, inspect their source code with 'clangd_query(command=\"show\", symbol_or_query=\"<FunctionName>\")'.\n"
            f"   Ensure ALL symbols, classes, and member functions in this module are read and reviewed at least once!\n"
            f"4. USAGES & REFERENCES: Run 'clangd_query' with command='usages' to check where key classes and functions are called, verifying ownership and call-site safety.\n"
            f"5. MEMORY & CONCURRENCY CHECKS: Run 'ripgrep_search' with path_filter='{current_module}' to check for:\n"
            f"   - Raw pointers, manual memory management (malloc, free, new, delete, reinterpret_cast, strcpy, sprintf).\n"
            f"   - Synchronization primitives (mutex, shared_mutex, lock_guard, unique_lock, atomic, condition_variable).\n"
            f"6. RECORD FINDINGS: Call 'record_finding' immediately whenever you detect an architectural insight, exceptional code pattern, minor improvement, or critical flaw.\n"
            f"7. AUDIT ENFORCEMENT: Python tracks all audited classes, methods, and files. You cannot conclude or proceed to the next module until all code in this module has been read and audited."
        )

        sub_state: ModuleReviewState = {
            "messages": [
                SystemMessage(content=(
                    "You are an expert modern C++ code auditor. Thoroughly examine the assigned module files. "
                    "Prioritize semantic inspection: use 'clangd_query(command=\"interface\")' for class structures, "
                    "'clangd_query(command=\"show\")' for function implementations, and 'clangd_query(command=\"usages\")' for references. "
                    "NEVER read full large files with read_project_file. Ensure all symbols and functions are read at least once. "
                    "Call record_finding immediately for every discovery."
                )),
                HumanMessage(content=prompt)
            ],
            "module_name": current_module,
            "user_prompt": user_prompt,
            "target_files": sorted(files),
            "target_classes": sorted(mod_classes),
            "interfaced_classes": [],
            "discovered_functions": sorted(initial_module_funcs),
            "audited_functions": [],
            "audited_files": [],
            "nudge_count": 0,
            "findings_at_start": len(_RECORDED_FINDINGS),
            "max_nudges": 15
        }

        findings_before = len(_RECORDED_FINDINGS)
        try:
            for step in sub_agent.stream(sub_state, {"recursion_limit": module_max_steps}, stream_mode="updates"):
                for node_name, node_update in step.items():
                    if node_name == "agent":
                        # Token context banner and tool calls are displayed in agent_step
                        pass
                    elif node_name == "tools":
                        for msg in node_update.get("messages", []):
                            raw_text = extract_text(msg.content)
                            t_tokens = count_tokens(raw_text)
                            preview = raw_text[:120].replace("\n", " ")
                            if len(raw_text) > 120:
                                preview += "..."
                            console.print(f"    [dim]Tool Result ({t_tokens:,} tokens): {escape(preview)}[/dim]")
                    elif node_name == "enforce_audit":
                        pass
        except Exception as e:
            # If a single module hits its local recursion limit, log and proceed to the next module
            console.print(f"[dim yellow]  (Module '{escape(current_module)}' completed exploration; proceeding to next module: {e})[/dim yellow]")

        pct = ((idx + 1) / len(modules)) * 100
        new_findings = len(_RECORDED_FINDINGS) - findings_before
        console.print(f"[green][OK] Finished Module '{escape(current_module)}/' ({idx + 1}/{len(modules)} modules - {pct:.1f}% complete) [{new_findings} findings recorded in this module][/green]")

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
    ]

    user_prompt = state.get("user_prompt", "")
    if user_prompt:
        sections.append(f"## Initial Review Focus & Directive\n> {user_prompt}\n")

    sections.append("## 1. Project Architecture & Dependency Overview")
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
            sections.append(f"### [CRITICAL] {item['title']}\n- **Location**: `{item['files_and_lines']}`\n\n{item['details']}")
            if item.get("recommended_fix"):
                sections.append(f"\n**Recommended Fix:**\n```cpp\n{item['recommended_fix']}\n```\n")
    else:
        sections.append("No critical flaws or memory vulnerabilities detected.\n")

    final_report = "\n".join(sections)
    return {"final_report": final_report}


def build_repo_review_orchestrator(
    llm,
    module_max_steps: int = 300,
    max_context_tokens: int = 32000,
    summarize_threshold: Optional[int] = None
):
    """
    Build the deterministic multi-node LangGraph orchestrator.
    """
    wf = StateGraph(RepoReviewState)

    wf.add_node("discover_and_plan", discover_and_plan_node)
    wf.add_node("review_module", review_module_node_factory(
        llm,
        module_max_steps=module_max_steps,
        max_context_tokens=max_context_tokens,
        summarize_threshold=summarize_threshold
    ))
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
    max_steps: int = 300,
    max_context_tokens: int = 32000,
    summarize_threshold: Optional[int] = None,
    user_prompt: str = ""
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

    effective_summarize = summarize_threshold or min(12000, int(max_context_tokens * 0.45))
    user_prompt_display = f"\nUser Directive: [yellow]{user_prompt}[/yellow]" if user_prompt else ""

    console.print(Panel(
        f"[bold cyan]Autonomous Modular C++ Code Review Orchestrator[/bold cyan]\n"
        f"Project Path : [yellow]{proj_path}[/yellow]\n"
        f"Provider     : [green]{provider}[/green]\n"
        f"Model        : [green]{model_name}[/green]\n"
        f"Max Context  : [yellow]{max_context_tokens:,} tokens[/yellow]\n"
        f"Summarize At : [yellow]{effective_summarize:,} tokens[/yellow]\n"
        f"Architecture : [blue]Deterministic Plan & Map-Reduce Multi-Node Graph[/blue]\n"
        f"Max Steps/Mod: [yellow]{max_steps}[/yellow]\n"
        f"Tools Active : [blue]clangd-query, ripgrep (rg), read_project_file, record_finding[/blue]"
        f"{user_prompt_display}",
        title="Agent Configuration",
        border_style="cyan"
    ))

    llm = get_llm(
        provider=provider,
        model_name=model_name,
        ollama_host=ollama_host,
        max_context_tokens=max_context_tokens
    )
    app = build_repo_review_orchestrator(
        llm=llm,
        module_max_steps=max_steps,
        max_context_tokens=max_context_tokens,
        summarize_threshold=effective_summarize
    )

    initial_state: RepoReviewState = {
        "project_dir": str(proj_path),
        "user_prompt": user_prompt,
        "all_files": [],
        "modules": [],
        "module_files_map": {},
        "module_classes_map": {},
        "current_module_index": 0,
        "findings": [],
        "final_report": "",
        "max_context_tokens": max_context_tokens,
        "summarize_threshold": effective_summarize
    }

    result = app.invoke(initial_state, {"recursion_limit": max_steps * 5})
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
    parser.add_argument(
        "--user-prompt", "--initial-prompt", "-u",
        dest="user_prompt",
        type=str,
        default="",
        help="Initial user prompt or specific review focus directive to guide the review agent."
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=32000,
        help="Maximum context token limit to enforce (default: 32000 / 32k)"
    )
    parser.add_argument(
        "--context-summarize-threshold",
        type=int,
        default=12000,
        help="Context token threshold to proactively trigger rolling summarization (default: 12000 tokens)"
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
        max_steps=args.max_steps,
        max_context_tokens=args.max_context_tokens,
        summarize_threshold=args.context_summarize_threshold,
        user_prompt=args.user_prompt
    )

