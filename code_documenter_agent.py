#!/usr/bin/env python3
"""
Autonomous C++ Codebase Documentation Agent using LangGraph, clangd-query, and ripgrep.
Explores the codebase folder-by-folder and incrementally writes a structured Markdown documentation file.
Enforces an active context size limit (defaulted to 32k tokens) to prevent context bloat and memory exhaustion.
"""

import os
import sys
import json
import time
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

# Shared C++ Analysis Tools
from cpp_agent_tools import (
    clangd_query,
    ripgrep_search,
    read_project_file,
    list_project_structure,
    set_active_project_dir,
    get_active_project_dir,
    ensure_compile_commands,
    get_llm,
    extract_text,
    discover_project_classes,
    group_classes_by_module
)

load_dotenv()
console = Console()

# ============================================================================
# Token Counting & Context Limiter (Default: 32k Tokens)
# ============================================================================

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
    # Fallback heuristic: roughly 3.8 characters per token for source code / english text
    return max(1, len(text) // 4)


def count_message_tokens(msg: BaseMessage) -> int:
    """Calculate token size of a LangChain message including tool calls."""
    tokens = count_tokens(extract_text(msg.content)) + 4  # message overhead
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

    # 1. Advance split_idx forward past any contiguous ToolMessages so we don't start recent_turns with an orphan ToolMessage
    while split_idx < len(messages) and isinstance(messages[split_idx], ToolMessage):
        split_idx += 1

    # 2. If the message immediately preceding split_idx was an AIMessage with tool_calls,
    # move split_idx back before that AIMessage so its tool call and response remain together.
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


def is_substantive_documentation(text: str) -> bool:
    """Check if text is an actual comprehensive documentation section rather than a conversational remark."""
    if not text or len(text.strip()) < 350:
        return False
    stripped = text.strip()
    has_markdown_structure = (
        "#" in stripped or "```" in stripped or "•" in stripped or "\n- " in stripped or "\n\n" in stripped
    )
    first_line = stripped.splitlines()[0].lower()
    is_transitional = any(first_line.startswith(prefix) for prefix in [
        "now i have", "i have thoroughly", "let me compile", "let me write", "i will now",
        "i am ready to", "let me proceed", "i will compile", "i will create", "i have now"
    ]) and len(stripped.splitlines()) < 4

    return has_markdown_structure and not is_transitional and len(stripped.splitlines()) >= 4


def manage_context_with_summarization(
    messages: List[BaseMessage],
    max_tokens: int = 32000,
    reserve_tokens: int = 2500,
    summarize_threshold: Optional[int] = None
) -> List[BaseMessage]:
    """
    Enforce strict context size limit with proactive rolling summarization.
    Never truncates active tool outputs to avoid information loss or code distortion.
    When cumulative conversation tokens exceed the threshold, safely summarizes completed older turns
    into a high-signal technical context summary while preserving recent active turns intact.
    """
    if summarize_threshold is None:
        summarize_threshold = min(12000, int(max_tokens * 0.45))
    effective_limit = min(summarize_threshold, max_tokens - reserve_tokens)

    cur_tokens = sum(count_message_tokens(m) for m in messages)

    if cur_tokens <= effective_limit or len(messages) <= 3:
        return messages

    # Context exceeds strict limit -> Perform Rolling Technical Summarization of older turns
    console.print(
        f"\n  [bold yellow][Notice] Context reached {cur_tokens:,} tokens (exceeding strict threshold {effective_limit:,}). "
        f"Active Rolling Summarization initiated...[/bold yellow]"
    )

    preserved_header = messages[:2] if len(messages) > 2 else messages
    conversation_tail = list(messages[2:]) if len(messages) > 2 else []

    history_to_summarize, recent_active_turns = partition_messages_safely(conversation_tail, target_recent_count=2)
    if not history_to_summarize:
        return messages

    # Extract exploration facts from history_to_summarize
    extracted_facts = []
    for m in history_to_summarize:
        if isinstance(m, ToolMessage):
            raw = extract_text(m.content).strip()
            if raw:
                preview = raw[:350].replace("\n", " ")
                extracted_facts.append(f"• Tool `{m.name}` revealed: {preview}")
        elif isinstance(m, AIMessage):
            text = extract_message_text(m)
            if text and len(text) > 30 and not is_substantive_documentation(text):
                extracted_facts.append(f"• Exploration Note: {text[:250].replace(chr(10), ' ')}")

    summary_text = (
        "### Summary of Prior Codebase Exploration (Condensed to stay within strict context limit):\n"
        + ("\n".join(extracted_facts[:15]) if extracted_facts else "Explored files and symbols in current module.")
    )

    summary_message = SystemMessage(content=summary_text)
    new_messages = preserved_header + [summary_message] + recent_active_turns
    new_tokens = sum(count_message_tokens(m) for m in new_messages)
    saved = cur_tokens - new_tokens
    console.print(f"  [bold green][OK] Context condensed from {cur_tokens:,} down to {new_tokens:,} tokens (-{(saved/max(1, cur_tokens))*100:.1f}%, saved {saved:,} tokens).[/bold green]\n")
    return new_messages


# Alias for backward compatibility
trim_messages_to_budget = manage_context_with_summarization


# ============================================================================
# Incremental Markdown Documentation Storage & Tools
# ============================================================================

_DOC_OUTPUT_FILE: Path = Path("CODEBASE_DOCUMENTATION.md")
_DOCUMENTED_SECTIONS: List[Dict[str, Any]] = []
_MAIN_SECTION_COUNTER: int = 0
_SUB_SECTION_COUNTER: int = 0


def init_documentation_file(output_path: Path, project_name: str) -> None:
    """Initialize the Markdown documentation file with title and metadata."""
    global _DOC_OUTPUT_FILE, _DOCUMENTED_SECTIONS, _MAIN_SECTION_COUNTER, _SUB_SECTION_COUNTER
    _DOC_OUTPUT_FILE = output_path.resolve()
    _DOCUMENTED_SECTIONS = []
    _MAIN_SECTION_COUNTER = 0
    _SUB_SECTION_COUNTER = 0

    header = (
        f"# {project_name} — Codebase Documentation\n\n"
        f"> *Generated incrementally by Autonomous C++ Codebase Documenter Agent*\n"
        f"> *Date: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        "<!-- TOC_PLACEHOLDER -->\n\n"
        "---\n\n"
    )
    with open(_DOC_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(header)
        f.flush()


def extract_section_summary(title: str, markdown_content: str) -> str:
    """
    Extract a structured, high-signal architectural summary from a section's Markdown documentation.
    Produces:
      - Functional Purpose & Responsibilities
      - Primary Classes, Structs, and Interfaces
      - Concurrency, Resource Management, and Key Mechanics
    """
    import re

    # 1. Strip code blocks, HTML comments, and markdown table markup
    clean_text = re.sub(r'<!--.*?-->', '', markdown_content, flags=re.DOTALL)
    text_without_code = re.sub(r'```.*?```', '', clean_text, flags=re.DOTALL).strip()

    # 2. Extract Key Classes / Interfaces / Structs (PascalCase symbols)
    symbols = set()
    for match in re.finditer(r'\b(?:class|struct|interface)\s+([A-Z][A-Za-z0-9_]{2,})\b', markdown_content):
        sym = match.group(1).strip()
        if sym not in {"Class", "Struct", "Interface", "Type", "Overview", "Architecture"}:
            symbols.add(sym)

    # Extract backticked identifiers in bullet points
    for match in re.finditer(r'[-*]\s+`([A-Za-z0-9_:]+)`', text_without_code):
        sym = match.group(1).strip()
        if len(sym) >= 3 and (sym[0].isupper() or "::" in sym or "_" in sym):
            if sym not in {"TODO", "NOTE", "WARNING", "IMPORTANT"}:
                symbols.add(sym)

    # 3. Extract opening functional overview / role narrative
    lines = [line.strip() for line in text_without_code.split("\n") if line.strip()]
    narrative_lines = []
    for line in lines:
        if line.startswith("#") or line.startswith("|") or line.startswith("---") or line.startswith(">") or line.startswith("<!--") or line.startswith("-") or line.startswith("*"):
            continue
        cleaned = re.sub(r'[`*_#]', '', line).strip()
        if len(cleaned) > 25:
            narrative_lines.append(cleaned)
            if len(" ".join(narrative_lines)) > 240:
                break

    overview_text = " ".join(narrative_lines)
    if not overview_text:
        overview_text = f"Functional implementation and architecture for {title}."

    if len(overview_text) > 300:
        cut = overview_text[:300].rsplit(".", 1)
        overview_text = (cut[0] + ".") if len(cut) > 1 and len(cut[0]) > 80 else overview_text[:280] + "..."

    # 4. Detect concurrency, threading, and resource management traits
    traits = []
    lower_content = markdown_content.lower()
    if "shared_mutex" in lower_content:
        traits.append("Read/Write Concurrency (`std::shared_mutex`)")
    elif "mutex" in lower_content or "lock_guard" in lower_content or "unique_lock" in lower_content:
        traits.append("Thread-Safe (`std::mutex` / RAII locks)")
    if "atomic" in lower_content:
        traits.append("Atomic State (`std::atomic`)")
    if "unique_ptr" in lower_content or "shared_ptr" in lower_content or "raii" in lower_content:
        traits.append("RAII Smart Pointer Ownership")
    if "socket" in lower_content or "epoll" in lower_content or "poll" in lower_content or "tcp" in lower_content:
        traits.append("Network I/O Handling")

    summary_lines = [
        f"• Purpose: {overview_text}"
    ]
    if symbols:
        sorted_syms = sorted(list(symbols))[:8]
        syms_str = ", ".join(f"`{s}`" for s in sorted_syms)
        summary_lines.append(f"• Key Types & Abstractions: {syms_str}")
    if traits:
        traits_str = ", ".join(traits)
        summary_lines.append(f"• Design & Concurrency: {traits_str}")

    return "\n".join(summary_lines)


def format_architectural_memory() -> str:
    """
    Generate a structured architectural memory ledger of all previously documented
    sections and modules, to be injected into the prompt of each subsequent module.
    """
    global _DOCUMENTED_SECTIONS
    if not _DOCUMENTED_SECTIONS:
        return "PRIOR ARCHITECTURAL CONTEXT: This is the first module being documented. No prior module documentation exists yet."

    lines = [
        "════════════════════════════════════════════════════════════════════════════════",
        "ARCHITECTURAL MEMORY OF PREVIOUSLY DOCUMENTED MODULES (Cross-Folder Context):",
        "Use this context to understand existing system abstractions, avoid duplicate explanations,",
        "and establish cross-module references and data-flow connections.",
        "════════════════════════════════════════════════════════════════════════════════"
    ]

    for sec in _DOCUMENTED_SECTIONS:
        lines.append(f"\n[{sec['title']}]")
        summary_body = sec.get("summary") or sec.get("preview", "")
        lines.append(summary_body)

    lines.append("\nCROSS-MODULE GUIDELINES FOR THIS FOLDER:")
    lines.append("• Connect this folder's components to the previously documented abstractions above.")
    lines.append("• If classes in this folder implement, inherit, call, or manage types from earlier sections, explain how they fit into the overall data and control flow.")
    lines.append("• Do NOT re-explain the internal details of already documented types; reference them concisely.")
    lines.append("════════════════════════════════════════════════════════════════════════════════")
    return "\n".join(lines)


@tool
def append_documentation_section(
    section_title: Optional[str] = None,
    markdown_content: Optional[str] = None,
    title: Optional[str] = None,
    content: Optional[str] = None,
    level: int = 2,
    **kwargs: Any
) -> str:
    """Incrementally write and append a new section to the codebase Markdown documentation file on disk.
    Call this tool as soon as you finish investigating a module, architectural component, or class hierarchy.

    Args:
      section_title: Semantic title for the section WITHOUT ANY SECTION NUMBERS (e.g. 'Order Processing & Payment Workflow', 'SessionManager & Memory Safety').
                     DO NOT include '1.' or 'Section 2' in the title; section numbering is automatically managed and sequenced for you.
      markdown_content: Comprehensive markdown text explaining functional behavior, business logic,
                        inner code mechanisms, public APIs, member variables, concurrency/thread-safety, and Mermaid diagrams.
      level: Header level (2 for main sections '##', 3 for subsections '###', default: 2)
    """
    global _DOC_OUTPUT_FILE, _DOCUMENTED_SECTIONS, _MAIN_SECTION_COUNTER, _SUB_SECTION_COUNTER

    import re

    raw_title = section_title or title or kwargs.get("heading") or kwargs.get("name") or "Component Architecture & Implementation"
    raw_content = markdown_content or content or kwargs.get("markdown") or kwargs.get("text") or ""
    if not raw_content or len(raw_content.strip()) < 30:
        return "Error: 'markdown_content' is required and must contain comprehensive markdown documentation text."

    # Strip any leading numbers, "Section X:", or Roman numerals the model may have generated
    clean_title = re.sub(r'^(?:Section\s*)?(?:\d+[\.\-_:]\s*)+', '', raw_title.strip(), flags=re.IGNORECASE).strip()
    clean_title = re.sub(r'^(?:Section\s*)?[IVXLCDM]+[\.\-_:]\s*', '', clean_title, flags=re.IGNORECASE).strip()
    if not clean_title:
        clean_title = raw_title.strip()

    # Automatically generate strict, monotonic, sequential numbering
    if level <= 2:
        _MAIN_SECTION_COUNTER += 1
        _SUB_SECTION_COUNTER = 0
        numbered_title = f"{_MAIN_SECTION_COUNTER}. {clean_title}"
    else:
        _SUB_SECTION_COUNTER += 1
        numbered_title = f"{_MAIN_SECTION_COUNTER}.{_SUB_SECTION_COUNTER} {clean_title}"

    prefix = "#" * max(1, min(level, 5))
    formatted_chunk = f"{prefix} {numbered_title}\n\n{raw_content.strip()}\n\n---\n\n"

    try:
        with open(_DOC_OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(formatted_chunk)
            f.flush()

        summary = extract_section_summary(numbered_title, raw_content)

        _DOCUMENTED_SECTIONS.append({
            "title": numbered_title,
            "raw_title": clean_title,
            "level": level,
            "timestamp": time.time(),
            "summary": summary,
            "preview": raw_content[:120].replace("\n", " ")
        })

        console.print(f"  [bold green][Appended Section (Level {level})]:[/bold green] [cyan]{escape(numbered_title)}[/cyan]")
        return f"Successfully appended section '{numbered_title}' ({len(raw_content)} chars) to documentation file. Total sections: {len(_DOCUMENTED_SECTIONS)}."
    except Exception as e:
        return f"Error writing to documentation file: {e}"


@tool
def read_current_documentation_toc() -> str:
    """Read the current Table of Contents and architectural summary of all sections that have already been written to disk.
    Use this to see what has already been documented, review existing abstractions, and avoid duplicate sections.
    """
    global _DOCUMENTED_SECTIONS
    if not _DOCUMENTED_SECTIONS:
        return "No documentation sections have been written yet."

    lines = ["Current Documented Sections & Architectural Memory:"]
    for i, s in enumerate(_DOCUMENTED_SECTIONS, 1):
        indent = "  " * max(0, (s.get("level", 2) - 1))
        lines.append(f"\n{indent}- **{s['title']}**")
        if "summary" in s and s["summary"]:
            for subline in s["summary"].split("\n"):
                lines.append(f"{indent}  {subline}")
    return "\n".join(lines)


EXPLORATION_TOOLS = [
    clangd_query,
    ripgrep_search,
    read_project_file,
    list_project_structure,
    read_current_documentation_toc
]

DOCUMENTER_TOOLS = EXPLORATION_TOOLS + [append_documentation_section]


# ============================================================================
# Documenter System Prompt
# ============================================================================

DOCUMENTER_SYSTEM_PROMPT = """You are an expert Principal C++ Software Architect and Technical Writer.
Your task is to generate clear, comprehensive, functional, publication-grade Markdown documentation for the given C++ codebase.

════════════════════════════════════════════════════════════════════════════════
CORE INSTRUCTIONS & STANDARDS:
1. INCREMENTAL DOCUMENTATION & CLEAN TITLES:
   - Document incrementally! Call `append_documentation_section` as you finish investigating each key component, file, or subsystem—do NOT wait until the very end to document everything at once.
   - For a module's main architecture overview, use level=2 ('##'). For individual files, classes, algorithms, or subsystems within the module, use level=3 ('###').
   - DO NOT provide section numbers in `section_title` (e.g. do NOT write 'Section 2' or '3. ...').
     Provide ONLY the descriptive semantic title (e.g. 'Order Processing Engine & Payment Workflow').
     Section numbers are automatically tracked and prefixed for you in sequence.
2. FOCUS ON FUNCTIONAL EXPLANATION & CODE INNER LOGIC:
   - Do NOT just produce a dry list of class names and field types.
   - Deeply explain WHAT the code does functionally: business logic, operational behavior, workflows, and state transitions.
   - EXPLAIN THE CRITICAL PARTS OF THE CODE:
     * Break down key algorithms and methods step-by-step.
     * Explain input validations, data transformations, error handling, return codes, and side effects.
     * Describe edge cases, failure recoveries, and performance considerations.
     * Explain dynamic interactions: trace how caller functions pass data into methods and what downstream effects occur.
3. SEMANTIC CODE INSPECTION OVER FULL-FILE READS:
   - AVOID FULL-FILE READS: NEVER use 'read_project_file' on large files (> 80 lines). Full file reads trigger severe context bloat and rate limits!
     Full-file reading is strictly restricted to small files (< 80 lines) or build configs (CMakeLists.txt).
   - Use 'clangd_query' with command='interface' and symbol_or_query='<ClassName>' to inspect class declarations, public interfaces, and member variables.
   - Use 'clangd_query' with command='show' and symbol_or_query='<ClassName::MethodName>' to inspect method bodies, inner logic, and step-by-step algorithms.
   - Use 'clangd_query' with command='usages' to inspect callers, references, and data-flow across the codebase.
   - Use 'ripgrep_search' with pattern='class ' or 'struct ' to discover all classes and structs in the codebase.
   - You MUST ensure all symbols, classes, and key functions in the assigned module are read and documented at least once!
   - Use 'ripgrep_search' to verify concurrency primitives ('mutex', 'shared_mutex', 'atomic') and resource ownership (smart pointers, RAII).
4. DIAGRAMS & FLOWCHARTS (MERMAID ONLY):
   - Whenever illustrating architecture, class relationships, state transitions, or execution flows, you MUST use Mermaid diagrams ONLY inside fenced code blocks (` ```mermaid ... ``` `).
   - Do NOT use ASCII art, plain text boxes, or pseudo-code drawings for diagrams.
   - Supported Mermaid types:
     * `flowchart TD` or `flowchart LR` for functional data/control pipelines.
     * `sequenceDiagram` for method call sequences and dynamic component interactions.
     * `classDiagram` for class inheritance and interface relationships.
   - Ensure valid Mermaid syntax: quote node labels containing special characters (parentheses, braces, brackets), e.g. `id["OrderRepository (Thread-Safe)"]`.
5. CONTEXT EFFICIENCY (32k LIMIT):
   - Keep tool queries targeted and focused.
   - Append sections incrementally to keep the context window compact and clean.
6. CROSS-MODULE ARCHITECTURAL MEMORY & CONTINUITY:
   - When moving between folders, you have access to the ARCHITECTURAL MEMORY of all previously documented modules.
   - Explicitly link your explanations to components established in prior sections (e.g. refer to classes/interfaces defined in header folders when analyzing implementation folders).
   - Avoid redundant duplication: build upon existing abstractions rather than re-explaining them from scratch.
════════════════════════════════════════════════════════════════════════════════
"""


# ============================================================================
# Multi-Node Documenter Graph
# ============================================================================

class DocumenterState(TypedDict):
    project_dir: str
    output_file: str
    user_prompt: str
    target_dirs: Optional[List[str]]
    ignore_dirs: Optional[List[str]]
    all_files: List[str]
    modules: List[str]
    module_files_map: Dict[str, List[str]]
    module_classes_map: Dict[str, List[str]]
    current_module_index: int
    sections_count: int
    max_context_tokens: int
    summarize_threshold: int


def doc_discover_and_plan_node(state: DocumenterState) -> Dict[str, Any]:
    """
    Node 1: Explore project directories, CMake configuration, and entry points.
    Plans documentation layout starting from the codebase entry point (main/bootstrap)
    and traversing topologically, initializing the output Markdown file.
    """
    import re
    proj_path = get_active_project_dir()
    out_file = Path(state["output_file"]).resolve()

    console.print("\n[bold cyan]═══ Phase 1: Codebase Discovery, Entry Point Detection & Documentation Planning ═══[/bold cyan]")

    user_prompt = state.get("user_prompt", "")
    if user_prompt:
        console.print(f"[cyan][User Documentation Directive]: {user_prompt}[/cyan]")

    init_documentation_file(out_file, proj_path.name)

    # Exclude build, cache, and user-specified ignored directories
    ignored = {
        "build", ".cache", ".git", ".vscode", ".idea",
        "third_party", "thirdparty", "external", "vendor",
        "deps", "_deps", "vcpkg_installed", "conan", "submodules"
    }
    user_ignored = state.get("ignore_dirs")
    if user_ignored:
        for ign in user_ignored:
            cleaned = ign.strip().lower().strip("/")
            if cleaned:
                ignored.add(cleaned)

    def is_ignored(p: Path) -> bool:
        for part in p.parts:
            part_lower = part.lower().strip()
            if part_lower in ignored or part_lower.startswith("."):
                return True
        posix = "/" + p.as_posix().lower() + "/"
        for ign in ignored:
            if ign and (f"/{ign}/" in posix or posix.startswith(f"/{ign}/") or posix.endswith(f"/{ign}/")):
                return True
        return False

    headers = [p.relative_to(proj_path) for p in sorted(proj_path.glob("**/*.h*")) if not is_ignored(p)]
    sources = [p.relative_to(proj_path) for p in sorted(proj_path.glob("**/*.c*")) if not is_ignored(p)]

    all_files = sorted(list(set([str(h) for h in headers] + [str(s) for s in sources])))

    # Discover classes and structs across project files
    file_to_classes = discover_project_classes(proj_path, ignored_dirs=ignored)
    module_classes_map = group_classes_by_module(file_to_classes)
    total_classes = sum(len(c) for c in module_classes_map.values())

    module_map: Dict[str, List[str]] = {}
    for f in all_files:
        parent = str(Path(f).parent)
        if parent not in module_map:
            module_map[parent] = []
        module_map[parent].append(f)

    sorted_modules = sorted(list(module_map.keys()))

    console.print(f"[green]Discovered {len(all_files)} total C++ files across {len(sorted_modules)} directories/modules.[/green]")
    console.print(f"[green]Discovered {total_classes} declared classes/structs across repository (scanned via rg class/struct).[/green]")

    # 1. Detect Entry Point (where application starts execution)
    entry_point_files: List[str] = []
    for s in sources:
        try:
            full_p = proj_path / s
            if full_p.exists():
                txt = full_p.read_text(encoding="utf-8", errors="ignore")
                if re.search(r'\bint\s+main\s*\(', txt) or re.search(r'\bvoid\s+main\s*\(', txt):
                    entry_point_files.append(str(s))
        except Exception:
            pass

    entry_point_module = None
    if entry_point_files:
        entry_point_module = str(Path(entry_point_files[0]).parent)
        console.print(f"\n[bold cyan][Starting Point Detected]:[/bold cyan] [green]{entry_point_files[0]}[/green] (Module: [yellow]{entry_point_module}/[/yellow])")
    else:
        for mod in sorted_modules:
            if "include" in mod.lower() or mod == ".":
                entry_point_module = mod
                break

    # 2. Check for target_dirs filter
    target_dirs = state.get("target_dirs")
    modules_to_document = sorted_modules
    if target_dirs:
        norm_targets = [t.strip().strip("/").lower() for t in target_dirs if t.strip()]
        filtered = []
        for m in sorted_modules:
            m_lower = m.strip().strip("/").lower()
            if any(m_lower == t or m_lower.startswith(t + "/") or t.startswith(m_lower + "/") for t in norm_targets):
                filtered.append(m)
        if filtered:
            modules_to_document = filtered
            console.print(f"\n[bold green]Target Scope Applied:[/bold green] Queued {len(modules_to_document)} module(s) under [{', '.join(norm_targets)}] for documentation:")
            for mod in modules_to_document:
                m_classes = module_classes_map.get(mod, [])
                cls_tag = f" — Classes: {', '.join(m_classes)}" if m_classes else ""
                console.print(f"  * [bold cyan]{escape(mod)}/[/bold cyan] ({len(module_map[mod])} files){cls_tag}")
            console.print("[dim]Note: Other folders can still be queried by clangd-query/rg for references if needed.[/dim]\n")
        else:
            console.print(f"[yellow]Warning: No modules matched target directories: {target_dirs}. Documenting all discovered modules.[/yellow]")

    # 3. Order modules topologically starting from the entry point
    def module_priority_sort(mod: str) -> int:
        if entry_point_module and mod == entry_point_module:
            return 0
        if entry_point_files:
            try:
                txt = (proj_path / entry_point_files[0]).read_text(encoding="utf-8", errors="ignore")
                for f in module_map.get(mod, []):
                    fname = Path(f).name
                    if f'"{fname}"' in txt or f'<{fname}>' in txt or fname in txt:
                        return 1
            except Exception:
                pass
        if "include" in mod.lower():
            return 2
        if "src" in mod.lower() or "source" in mod.lower():
            return 3
        return 4

    modules_to_document = sorted(modules_to_document, key=lambda m: (module_priority_sort(m), m))

    console.print(f"\n[bold green]Determined Documentation Order ({len(modules_to_document)} modules):[/bold green]")
    for rank, mod in enumerate(modules_to_document, 1):
        tag = " [bold magenta][Entry Point][/bold magenta]" if (entry_point_module and mod == entry_point_module) else ""
        m_classes = module_classes_map.get(mod, [])
        cls_tag = f" — Classes: {', '.join(m_classes)}" if m_classes else ""
        console.print(f"  {rank}. [bold cyan]{escape(mod)}/[/bold cyan] ({len(module_map[mod])} files){cls_tag}{tag}")

    # Read CMakeLists.txt to build Section 1
    cmakelists = proj_path / "CMakeLists.txt"
    cmake_text = ""
    if cmakelists.exists():
        try:
            with open(cmakelists, "r", encoding="utf-8") as f:
                cmake_text = f.read()
        except Exception:
            pass

    # Append Section 1: Architecture & Project Structure (title has NO number; numbering is automatic)
    sec1_content = (
        f"### Overview\n"
        f"This repository contains **{len(all_files)} C++ files** organized across **{len(sorted_modules)} directories**.\n"
    )
    if user_prompt:
        sec1_content += f"> **User Initial Documentation Directive**: {user_prompt}\n\n"

    if entry_point_files:
        sec1_content += f"> **Primary Application Entry Point**: `{entry_point_files[0]}` (Module `{entry_point_module}/`)\n\n"

    if target_dirs and len(modules_to_document) < len(sorted_modules):
        sec1_content += (
            f"> **Documentation Scope**: Detailed functional documentation is generated specifically for: `{', '.join(target_dirs)}` "
            f"({len(modules_to_document)} target modules). External and vendor folders are referenced where needed but omitted from dedicated chapters.\n\n"
        )
    else:
        sec1_content += "\n"

    sec1_content += "### Directory & Module Layout\n"
    for mod in sorted_modules:
        marker = "*(Documented)*" if mod in modules_to_document else "*(Reference)*"
        m_classes = module_classes_map.get(mod, [])
        cls_info = f" (Classes: {', '.join(m_classes)})" if m_classes else ""
        sec1_content += f"- **`{mod}/`** ({len(module_map[mod])} files){cls_info} {marker}:\n"
        for f in module_map[mod][:8]:
            sec1_content += f"  - `{Path(f).name}`\n"
        if len(module_map[mod]) > 8:
            sec1_content += f"  - *... and {len(module_map[mod]) - 8} more files*\n"

    if cmake_text:
        sec1_content += f"\n### Build System & Configuration (`CMakeLists.txt`)\n```cmake\n{cmake_text[:800]}\n```\n"

    append_documentation_section.invoke({
        "section_title": "System Architecture & Build Configuration",
        "markdown_content": sec1_content,
        "level": 2
    })

    return {
        "all_files": all_files,
        "modules": modules_to_document,
        "module_files_map": module_map,
        "module_classes_map": module_classes_map,
        "current_module_index": 0,
        "sections_count": 1,
        "user_prompt": user_prompt
    }


def message_reducer(existing: List[BaseMessage], update: Any) -> List[BaseMessage]:
    """Custom reducer supporting list concatenation and explicit context override upon summarization."""
    if isinstance(update, tuple) and len(update) == 2 and update[0] == "override":
        return list(update[1])
    if isinstance(update, list):
        return list(existing) + list(update)
    if isinstance(update, BaseMessage):
        return list(existing) + [update]
    return list(existing)


class ModuleAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], message_reducer]
    module_name: str
    append_called: bool
    append_count: int
    uncommitted_explorations: int
    reminder_count: int


def build_module_documenter_runner(
    llm,
    max_context_tokens: int = 32000,
    summarize_threshold: Optional[int] = None
):
    """
    Build focused LangGraph sub-agent for documenting a specific directory module,
    strictly bounded by proactive context limits, incremental in-module documentation,
    and guaranteed synthesis.
    """
    llm_with_tools = llm.bind_tools(DOCUMENTER_TOOLS)

    def agent_step(state: ModuleAgentState) -> Dict[str, Any]:
        trimmed_messages = manage_context_with_summarization(
            state["messages"],
            max_tokens=max_context_tokens,
            summarize_threshold=summarize_threshold
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
                    match = re.search(r"retry in (\d+(?:\d+)?)s", err_msg, re.IGNORECASE)
                    wait_time = (float(match.group(1)) + 2) if match else base_delay * (2 ** (attempt - 1))
                    console.print(f"[yellow]Rate limit (429). Waiting {wait_time:.1f}s (attempt {attempt}/{max_retries})...[/yellow]")
                    time.sleep(wait_time)
                else:
                    raise e

        # Calculate exact context size including the model's generated response
        total_tokens = sum(count_message_tokens(m) for m in trimmed_messages) + count_message_tokens(response)
        pct = (total_tokens / max(1, max_context_tokens)) * 100
        color = "green" if pct < 40 else ("yellow" if pct < 75 else "bold red")

        # Explicitly print context size before every tool call as requested
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

    def route_module_agent_step(state: ModuleAgentState) -> str:
        last_msg = state["messages"][-1]

        # 1. Did the agent invoke append_documentation_section?
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            has_append = any(tc.get("name") == "append_documentation_section" for tc in last_msg.tool_calls)
            if has_append:
                return "execute_append"

            # Agent wants to run exploration tools (read_project_file, clangd_query, ripgrep_search)
            uncommitted = state.get("uncommitted_explorations", 0)
            if uncommitted >= 2:
                # Enforce documenting what was just explored before reading further files
                return "enforce_append_now"

            return "tools"

        # 2. No tool calls:
        if state.get("append_called", False):
            return END

        txt = extract_message_text(last_msg)
        # Only commit directly if it is ACTUAL comprehensive documentation
        if is_substantive_documentation(txt):
            return "commit_text_as_section"

        # If it's a transitional statement ("Now I will compile...") or intermediate thought:
        reminders = state.get("reminder_count", 0)
        if reminders < 1:
            return "remind_to_append"
        else:
            # Reached end of exploration or repeated conversational text -> trigger synthesis!
            return "synthesize_and_append"

    exploration_tool_node = ToolNode(EXPLORATION_TOOLS, handle_tool_errors=True)

    def execute_exploration_tools_node(state: ModuleAgentState) -> Dict[str, Any]:
        res = exploration_tool_node.invoke(state)
        return {
            "messages": res.get("messages", []),
            "uncommitted_explorations": state.get("uncommitted_explorations", 0) + 1
        }

    def enforce_append_now_node(state: ModuleAgentState) -> Dict[str, Any]:
        mod = state.get("module_name", "Module")
        notice = (
            f"[MANDATORY DOCUMENTATION CADENCE: You have explored code in module '{mod}'. "
            f"To document incrementally as requested and prevent context overflow, you must NOT inspect further files right now. "
            f"Invoke 'append_documentation_section' NOW to document the functional role, classes, inner algorithms, "
            f"and concurrency of the component you just inspected before continuing!]"
        )
        return {
            "messages": [HumanMessage(content=notice)],
            "uncommitted_explorations": 0
        }

    def execute_append_node(state: ModuleAgentState) -> Dict[str, Any]:
        last_msg = state["messages"][-1]
        tool_messages = []
        appends_in_this_step = 0
        mod = state.get("module_name", "Module")
        prior_appends = state.get("append_count", 0)

        for tc in getattr(last_msg, "tool_calls", []):
            if tc.get("name") == "append_documentation_section":
                args = dict(tc.get("args", {}))
                # Automatic subsection nesting inside module:
                total_appends = prior_appends + appends_in_this_step
                if total_appends > 0 and args.get("level", 2) <= 2:
                    args["level"] = 3
                try:
                    res = append_documentation_section.invoke(args)
                except Exception as e:
                    res = f"Error executing append_documentation_section: {e}"
                appends_in_this_step += 1
                tool_messages.append(ToolMessage(
                    content=(
                        f"{res}\n"
                        f"[Next Step: You can continue exploring and documenting other files or subsystems in module '{mod}' "
                        f"by calling 'read_project_file' or 'append_documentation_section'. When all files/components in '{mod}' "
                        f"have been documented, reply stating that documentation for this module is complete (with no further tool calls).]"
                    ),
                    tool_call_id=tc.get("id", "append_id"),
                    name="append_documentation_section"
                ))
            else:
                for t in EXPLORATION_TOOLS:
                    if t.name == tc.get("name"):
                        try:
                            res = t.invoke(tc.get("args", {}))
                        except Exception as e:
                            res = f"Error executing tool {t.name}: {e}"
                        tool_messages.append(ToolMessage(
                            content=str(res),
                            tool_call_id=tc.get("id", "tool_id"),
                            name=t.name
                        ))
        return {
            "messages": tool_messages,
            "append_called": True,
            "append_count": prior_appends + appends_in_this_step,
            "uncommitted_explorations": 0
        }

    def remind_to_append_node(state: ModuleAgentState) -> Dict[str, Any]:
        reminders = state.get("reminder_count", 0) + 1
        mod = state.get("module_name", "Module")

        reminder_text = (
            f"You have explored the code for module '{mod}'.\n"
            f"Now generate the functional Markdown documentation section for module '{mod}'.\n"
            f"Detail what the code does functionally, its critical classes/methods, concurrency synchronization, and a Mermaid diagram.\n"
            f"You can either invoke 'append_documentation_section' or output the complete Markdown documentation text now."
        )

        return {
            "messages": [HumanMessage(content=reminder_text)],
            "reminder_count": reminders
        }

    def synthesize_and_append_node(state: ModuleAgentState) -> Dict[str, Any]:
        mod = state.get("module_name", "Module")
        clean_title = f"{mod.replace('/', ' ').title()} - Functional Architecture & Implementation"

        console.print(f"  [bold yellow][Notice] Compiling full publication-grade documentation for module '{escape(mod)}'...[/bold yellow]")

        synth_instruction = (
            f"You have completed your code exploration of module '{mod}'.\n"
            f"Now generate the complete, comprehensive, publication-grade functional Markdown documentation for module '{mod}'.\n\n"
            f"MANDATORY DOCUMENTATION REQUIREMENTS:\n"
            f"1. **Role & Functional Overview**: Deeply explain what this module actually does in practice, its business logic, operational behavior, and system role.\n"
            f"2. **Critical Code & Logic Breakdown**: Detail the key classes, member variables, algorithms, functions, input parameters, concurrency synchronization (mutex/shared_mutex), and error handling.\n"
            f"3. **Architectural / Flow Diagram**: Include at least one Mermaid diagram (```mermaid ... ```, strictly Mermaid only, no ASCII art) showing component interactions or runtime execution flow.\n"
            f"4. **Cross-Module Integration**: Explain how this module interacts with previously documented modules.\n\n"
            f"Produce the complete Markdown documentation text now."
        )

        synth_messages = manage_context_with_summarization(
            state["messages"] + [HumanMessage(content=synth_instruction)],
            max_tokens=max_context_tokens,
            summarize_threshold=summarize_threshold
        )
        print_context_banner(synth_messages, max_tokens=max_context_tokens, stage="Generating Full Documentation")

        try:
            resp = llm.invoke(synth_messages)
            doc_content = extract_message_text(resp)
        except Exception as e:
            console.print(f"[yellow]Synthesis invocation error: {e}[/yellow]")
            doc_content = ""

        if not is_substantive_documentation(doc_content):
            console.print("  [dim yellow](Model returned short text; prompting directly for full markdown breakdown...)[/dim yellow]")
            try:
                resp2 = llm.invoke([
                    SystemMessage(content=DOCUMENTER_SYSTEM_PROMPT),
                    HumanMessage(content=synth_instruction)
                ])
                doc_content = extract_message_text(resp2)
            except Exception:
                pass

        if not doc_content or len(doc_content.strip()) < 100:
            doc_content = (
                f"### Functional Architecture of `{mod}/`\n\n"
                f"Module `{mod}` provides core functionality explored during codebase analysis.\n\n"
                f"```mermaid\ngraph TD\n    Mod[\"{mod} Module\"] --> Sub[\"Core Implementation\"]\n```\n"
            )

        append_documentation_section.invoke({
            "section_title": clean_title,
            "markdown_content": doc_content,
            "level": 2
        })
        return {"append_called": True, "append_count": state.get("append_count", 0) + 1, "uncommitted_explorations": 0}

    def commit_text_as_section_node(state: ModuleAgentState) -> Dict[str, Any]:
        mod = state.get("module_name", "Module")
        clean_title = f"{mod.replace('/', ' ').title()} - Functional Architecture & Implementation"

        # Search backward for the most comprehensive assistant response
        candidates = []
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage):
                txt = extract_message_text(m)
                if is_substantive_documentation(txt):
                    candidates.append(txt)

        doc_content = candidates[0] if candidates else ""
        if not doc_content:
            return synthesize_and_append_node(state)

        console.print(f"  [bold green][Notice] Saving complete generated markdown documentation ({len(doc_content)} chars)...[/bold green]")
        append_documentation_section.invoke({
            "section_title": clean_title,
            "markdown_content": doc_content,
            "level": 2
        })
        return {"append_called": True, "append_count": state.get("append_count", 0) + 1, "uncommitted_explorations": 0}

    wf = StateGraph(ModuleAgentState)
    wf.add_node("agent", agent_step)
    wf.add_node("tools", execute_exploration_tools_node)
    wf.add_node("enforce_append_now", enforce_append_now_node)
    wf.add_node("execute_append", execute_append_node)
    wf.add_node("remind_to_append", remind_to_append_node)
    wf.add_node("synthesize_and_append", synthesize_and_append_node)
    wf.add_node("commit_text_as_section", commit_text_as_section_node)

    wf.add_edge(START, "agent")
    wf.add_conditional_edges("agent", route_module_agent_step, {
        "tools": "tools",
        "enforce_append_now": "enforce_append_now",
        "execute_append": "execute_append",
        "remind_to_append": "remind_to_append",
        "synthesize_and_append": "synthesize_and_append",
        "commit_text_as_section": "commit_text_as_section",
        END: END
    })
    wf.add_edge("tools", "agent")
    wf.add_edge("enforce_append_now", "agent")
    wf.add_edge("remind_to_append", "agent")
    wf.add_edge("synthesize_and_append", END)
    wf.add_edge("execute_append", "agent")
    wf.add_edge("commit_text_as_section", END)

    return wf.compile()


def extract_message_text(msg: Any) -> str:
    """Extract plain text from an AIMessage, content list, or string, with fallback to tool call args."""
    if msg is None:
        return ""
    if isinstance(msg, str):
        return msg.strip()

    content = getattr(msg, "content", msg)
    if isinstance(content, str) and content.strip():
        return content.strip()
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and item["text"]:
                    parts.append(str(item["text"]))
                elif "content" in item and item["content"]:
                    parts.append(str(item["content"]))
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text", "")))
            elif hasattr(item, "content"):
                parts.append(str(getattr(item, "content", "")))
        combined = "".join(parts).strip()
        if combined:
            return combined

    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            args = tc.get("args", {})
            if "markdown_content" in args and str(args["markdown_content"]).strip():
                return str(args["markdown_content"]).strip()

    return ""


def document_module_node_factory(
    llm,
    max_context_tokens: int = 32000,
    module_max_steps: int = 50,
    summarize_threshold: Optional[int] = None
):
    """
    Create document_module node with 32k context limitation, step control, functional focus,
    and guaranteed section appending.
    """
    sub_agent = build_module_documenter_runner(
        llm,
        max_context_tokens=max_context_tokens,
        summarize_threshold=summarize_threshold
    )

    def document_module_node(state: DocumenterState) -> Dict[str, Any]:
        idx = state["current_module_index"]
        modules = state["modules"]
        current_module = modules[idx]
        files = state["module_files_map"].get(current_module, [])

        console.print(f"\n[bold yellow]═══ Phase 2: Documenting Module ({idx + 1}/{len(modules)}): [cyan]{escape(current_module)}/[/cyan] ({len(files)} files) ═══[/bold yellow]")

        arch_memory = format_architectural_memory()

        # Discovered classes for this module
        mod_classes = state.get("module_classes_map", {}).get(current_module, [])
        classes_str = ", ".join(f"`{c}`" for c in mod_classes) if mod_classes else "None directly declared (audit member/free functions)"

        user_prompt_section = ""
        user_prompt = state.get("user_prompt", "")
        if user_prompt:
            user_prompt_section = (
                f"════════════════════════════════════════════════════════════════════════════════\n"
                f"USER INITIAL DOCUMENTATION DIRECTIVE (HIGH PRIORITY FOCUS):\n"
                f"{user_prompt}\n"
                f"Ensure your documentation specifically addresses and emphasizes this directive.\n"
                f"════════════════════════════════════════════════════════════════════════════════\n\n"
            )

        prompt = (
            f"You are writing in-depth functional C++ documentation for module '{current_module}'.\n"
            f"Files in this module:\n" + "\n".join(f"- {f}" for f in files) + "\n\n"
            f"Known Classes & Structs in this module (discovered via 'rg class/struct'):\n{classes_str}\n\n"
            f"{user_prompt_section}"
            f"{arch_memory}\n\n"
            f"MANDATORY CODE INSPECTION & DOCUMENTATION PROTOCOL:\n"
            f"1. AVOID FULL-FILE READS: NEVER use 'read_project_file' on large files (> 80 lines). Full file reads trigger severe context bloat! "
            f"Full-file reading is strictly restricted to small files (< 80 lines) or build configs (CMakeLists.txt).\n"
            f"2. SEMANTIC CLASS INSPECTION:\n"
            f"   - For each class/struct in this module ({classes_str}), call 'clangd_query(command=\"interface\", symbol_or_query=\"<ClassName>\")' "
            f"to inspect public interfaces, member variables, and methods.\n"
            f"   - For member function bodies, algorithms, and inner logic, call 'clangd_query(command=\"show\", symbol_or_query=\"<ClassName::MethodName>\")'.\n"
            f"   - For cross-file dependencies and callers, call 'clangd_query(command=\"usages\", symbol_or_query=\"<SymbolName>\")'.\n"
            f"3. FUNCTION AUDIT & ENSURE ALL SYMBOLS INSPECTED:\n"
            f"   - For standalone functions, inspect with 'clangd_query(command=\"show\", symbol_or_query=\"<FunctionName>\")'.\n"
            f"   - Ensure ALL symbols, classes, and member functions in this module are read and documented at least once!\n"
            f"   - Use 'ripgrep_search' with pattern='class ' or 'struct ' to discover any additional unmapped types.\n"
            f"4. INCREMENTAL DOCUMENTATION PROTOCOL:\n"
            f"   - Document incrementally! Call 'append_documentation_section' after inspecting each component/class hierarchy.\n"
            f"   - Do NOT read all files first; document as you go.\n"
            f"5. MERMAID DIAGRAMS:\n"
            f"   - Include Mermaid diagrams (strictly inside ```mermaid ... ```, no ASCII art) showing component interactions.\n"
            f"6. CONCLUDE: When all files/components in '{current_module}' have been documented, reply stating documentation is complete.\n\n"
            f"Begin by inspecting the first class or structure using 'clangd_query' or 'read_project_file'."
        )

        sub_state: ModuleAgentState = {
            "messages": [
                SystemMessage(content=DOCUMENTER_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ],
            "module_name": current_module,
            "append_called": False,
            "append_count": 0,
            "uncommitted_explorations": 0,
            "reminder_count": 0
        }

        sections_before = len(_DOCUMENTED_SECTIONS)

        try:
            for step in sub_agent.stream(sub_state, {"recursion_limit": module_max_steps}, stream_mode="updates"):
                for node_name, node_update in step.items():
                    if node_name == "agent":
                        # Context size banner and tool calls are displayed in agent_step
                        pass
                    elif node_name == "tools":
                        for msg in node_update.get("messages", []):
                            raw_text = extract_text(msg.content)
                            t_tokens = count_tokens(raw_text)
                            preview = raw_text[:120].replace("\n", " ")
                            if len(raw_text) > 120:
                                preview += "..."
                            console.print(f"    [dim]Tool Result ({t_tokens:,} tokens): {escape(preview)}[/dim]")
                    elif node_name == "enforce_append_now":
                        console.print("  [yellow][Notice] Prompting agent to document current concept before reading more files...[/yellow]")
                    elif node_name == "remind_to_append":
                        console.print("  [yellow][Notice] Nudging agent to invoke 'append_documentation_section'...[/yellow]")
                    elif node_name == "execute_append":
                        console.print("  [bold green][OK] Section committed to documentation file.[/bold green]")
                    elif node_name == "commit_text_as_section":
                        console.print("  [bold green][OK] Saved agent's markdown documentation to file.[/bold green]")
                    elif node_name == "synthesize_and_append":
                        console.print("  [bold green][OK] Saved guaranteed synthesized documentation to file.[/bold green]")
        except Exception as e:
            console.print(f"[dim yellow]  (Module exploration step ended: {e})[/dim yellow]")

        # In the unlikely event that an external error aborted execution before append was called:
        if len(_DOCUMENTED_SECTIONS) == sections_before:
            clean_module_title = f"{current_module.replace('/', ' ').title()} - Overview & Architecture"
            basic_doc = (
                f"### Architectural Overview of `{current_module}/`\n\n"
                f"This module contains {len(files)} C++ source and header file(s):\n"
                + "\n".join(f"- `{f}`" for f in files) + "\n\n"
                f"```mermaid\n"
                f"graph TD\n"
                f"    Module[\"{current_module} Module\"]\n"
                + "\n".join(f"    Module --> F{i}[\"{Path(f).name}\"]" for i, f in enumerate(files[:10]))
                + "\n```\n"
            )
            append_documentation_section.invoke({
                "section_title": clean_module_title,
                "markdown_content": basic_doc,
                "level": 2
            })

        pct = ((idx + 1) / len(modules)) * 100
        console.print(f"[green][OK] Finished Module '{escape(current_module)}/' ({idx + 1}/{len(modules)} modules - {pct:.1f}% complete)[/green]")

        return {
            "current_module_index": idx + 1,
            "sections_count": len(_DOCUMENTED_SECTIONS)
        }

    return document_module_node


def should_continue_documentation(state: DocumenterState) -> str:
    """Check whether any modules remain to be documented."""
    if state["current_module_index"] < len(state["modules"]):
        return "document_module"
    return "finalize_documentation"


def finalize_documentation_node(state: DocumenterState) -> Dict[str, Any]:
    """
    Node 3: Finalize documentation, generate clean Table of Contents,
    and update the markdown file header.
    """
    global _DOC_OUTPUT_FILE, _DOCUMENTED_SECTIONS
    console.print("\n[bold cyan]═══ Phase 3: Finalizing Table of Contents & Documentation Index ═══[/bold cyan]")

    out_file = Path(state["output_file"]).resolve()

    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Build Table of Contents
            toc_lines = ["## Table of Contents\n"]
            for i, s in enumerate(_DOCUMENTED_SECTIONS, 1):
                indent = "  " * max(0, s.get("level", 2) - 2)
                # Create anchor slug
                slug = s["title"].lower().replace(" ", "-").replace(".", "").replace("/", "").replace("`", "")
                toc_lines.append(f"{indent}- [{s['title']}](#{slug})")

            toc_text = "\n".join(toc_lines) + "\n"

            if "<!-- TOC_PLACEHOLDER -->" in content:
                content = content.replace("<!-- TOC_PLACEHOLDER -->", toc_text)
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
        except Exception as e:
            console.print(f"[yellow]Warning updating TOC: {e}[/yellow]")

    console.print(f"[bold green][OK] Documentation complete![/bold green] Total sections documented: {len(_DOCUMENTED_SECTIONS)}")
    console.print(f"[bold green]Saved to:[/bold green] [cyan]{out_file}[/cyan]\n")
    return {}


def build_codebase_documenter_graph(
    llm,
    max_context_tokens: int = 32000,
    module_max_steps: int = 50,
    summarize_threshold: Optional[int] = None
):
    """
    Build the deterministic multi-node documentation orchestrator.
    """
    wf = StateGraph(DocumenterState)

    wf.add_node("discover_and_plan", doc_discover_and_plan_node)
    wf.add_node(
        "document_module",
        document_module_node_factory(
            llm,
            max_context_tokens=max_context_tokens,
            module_max_steps=module_max_steps,
            summarize_threshold=summarize_threshold
        )
    )
    wf.add_node("finalize_documentation", finalize_documentation_node)

    wf.add_edge(START, "discover_and_plan")
    wf.add_edge("discover_and_plan", "document_module")
    wf.add_conditional_edges("document_module", should_continue_documentation, {
        "document_module": "document_module",
        "finalize_documentation": "finalize_documentation"
    })
    wf.add_edge("finalize_documentation", END)

    return wf.compile()


# ============================================================================
# Main Execution Runner
# ============================================================================

def run_codebase_documenter(
    project_dir: str,
    output_path: Optional[str] = None,
    provider: str = "gemini",
    model_name: Optional[str] = None,
    ollama_host: str = "http://localhost:11434",
    max_context_tokens: int = 32000,
    summarize_threshold: int = 12000,
    module_max_steps: int = 50,
    target_dirs: Optional[List[str]] = None,
    ignore_dirs: Optional[List[str]] = None
) -> Path:
    """
    Execute the autonomous codebase documentation agent loop.
    """
    proj_path = Path(project_dir).resolve()
    if not proj_path.exists() or not proj_path.is_dir():
        console.print(f"[bold red]Error: Project directory '{project_dir}' does not exist.[/bold red]")
        sys.exit(1)

    set_active_project_dir(proj_path)
    ensure_compile_commands(proj_path)

    if not output_path:
        out_file = proj_path / "CODEBASE_DOCUMENTATION.md"
    else:
        out_file = Path(output_path).resolve()

    if not model_name:
        if provider.lower() == "ollama":
            model_name = "llama3.1:8b"
        else:
            model_name = "gemini-3.5-flash-lite"

    target_display = f"[cyan]{', '.join(target_dirs)}[/cyan]" if target_dirs else "[yellow]All repository modules[/yellow]"
    ignore_display = f"[red]{', '.join(ignore_dirs)}[/red]" if ignore_dirs else "[dim]Standard build/vendor artifacts[/dim]"
    user_prompt_display = f"\nUser Directive  : [yellow]{user_prompt}[/yellow]" if user_prompt else ""

    console.print(Panel(
        f"[bold cyan]Autonomous C++ Codebase Documentation Agent[/bold cyan]\n"
        f"Project Path    : [yellow]{proj_path}[/yellow]\n"
        f"Target Scope    : {target_display}\n"
        f"Ignored Folders : {ignore_display}\n"
        f"Output Document : [green]{out_file}[/green]\n"
        f"Provider        : [green]{provider}[/green]\n"
        f"Model           : [green]{model_name}[/green]\n"
        f"Context Limit   : [magenta]{max_context_tokens:,} tokens (32k window guard)[/magenta]\n"
        f"Summarize Limit : [magenta]{summarize_threshold:,} tokens (proactive compression)[/magenta]\n"
        f"Steps / Module  : [yellow]{module_max_steps}[/yellow]\n"
        f"Tools Active    : [blue]clangd-query, ripgrep (rg), read_project_file, append_documentation_section[/blue]"
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
    app = build_codebase_documenter_graph(
        llm=llm,
        max_context_tokens=max_context_tokens,
        module_max_steps=module_max_steps,
        summarize_threshold=summarize_threshold
    )

    initial_state: DocumenterState = {
        "project_dir": str(proj_path),
        "output_file": str(out_file),
        "user_prompt": user_prompt,
        "target_dirs": target_dirs,
        "ignore_dirs": ignore_dirs,
        "all_files": [],
        "modules": [],
        "module_files_map": {},
        "module_classes_map": {},
        "current_module_index": 0,
        "sections_count": 0,
        "max_context_tokens": max_context_tokens,
        "summarize_threshold": summarize_threshold
    }

    app.invoke(initial_state, {"recursion_limit": 500})
    return out_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Autonomous C++ Codebase Documentation Agent with 32k context limit, target folder scoping, and incremental Markdown generation."
    )
    parser.add_argument(
        "--project-dir", "-p",
        type=str,
        default="./sample_project",
        help="Path to the C++ project directory (default: ./sample_project)"
    )
    parser.add_argument(
        "--target-dirs", "--include-dirs",
        type=str,
        default=None,
        help="Comma-separated list of folders/directories to generate documentation for (e.g. 'src,include' or 'src/engine'). "
             "Other folders (such as third_party or vendor) can still be explored and referenced by tools, but won't be documented."
    )
    parser.add_argument(
        "--ignore-dirs",
        type=str,
        default=None,
        help="Comma-separated list of folder/subfolder names to ignore completely during documentation (e.g. 'tests,benchmarks,legacy')."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output Markdown documentation file path (default: <project-dir>/CODEBASE_DOCUMENTATION.md)"
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
        help="Model name (e.g. 'gemini-3.5-flash-lite', 'llama3.1:8b', 'qwen2.5:14b')"
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama host URL (default: http://localhost:11434)"
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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum tool steps per module exploration (default: 50)"
    )
    parser.add_argument(
        "--user-prompt", "--initial-prompt", "-u",
        dest="user_prompt",
        type=str,
        default="",
        help="Initial user prompt or specific documentation focus directive to guide the documenter agent."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    targets = None
    if args.target_dirs:
        targets = [d.strip() for d in args.target_dirs.split(",") if d.strip()]

    ignores = None
    if args.ignore_dirs:
        ignores = [d.strip() for d in args.ignore_dirs.split(",") if d.strip()]

    run_codebase_documenter(
        project_dir=args.project_dir,
        output_path=args.output,
        provider=args.provider,
        model_name=args.model,
        ollama_host=args.ollama_host,
        max_context_tokens=args.max_context_tokens,
        summarize_threshold=args.context_summarize_threshold,
        module_max_steps=args.max_steps,
        target_dirs=targets,
        ignore_dirs=ignores,
        user_prompt=args.user_prompt
    )
