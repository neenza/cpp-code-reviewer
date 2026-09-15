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
    extract_text
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


def trim_messages_to_budget(
    messages: List[BaseMessage],
    max_tokens: int = 32000,
    reserve_tokens: int = 2500
) -> List[BaseMessage]:
    """
    Enforce strict context size limit (default 32k).
    Preserves SystemMessage (index 0) and task prompt (index 1).
    If total tokens exceed max_tokens - reserve_tokens, trims or truncates older tool messages
    and intermediate turns from the beginning while preserving valid tool-call pairing.
    """
    allowed_budget = max_tokens - reserve_tokens
    total_tokens = sum(count_message_tokens(m) for m in messages)

    if total_tokens <= allowed_budget:
        return messages

    # We need to trim. Always preserve system message (0) and user initial prompt (1)
    if len(messages) <= 2:
        return messages

    preserved_header = messages[:2]
    conversation_tail = messages[2:]

    # First pass: truncate excessively large tool message contents in the tail
    modified_tail: List[BaseMessage] = []
    for msg in conversation_tail:
        if isinstance(msg, ToolMessage):
            content_str = extract_text(msg.content)
            if count_tokens(content_str) > 800:
                truncated = content_str[:2500] + "\n... [Output truncated to respect 32k context budget]"
                new_msg = ToolMessage(content=truncated, tool_call_id=msg.tool_call_id, name=msg.name)
                modified_tail.append(new_msg)
            else:
                modified_tail.append(msg)
        else:
            modified_tail.append(msg)

    # Re-check tokens
    cur_tokens = sum(count_message_tokens(m) for m in preserved_header) + sum(count_message_tokens(m) for m in modified_tail)
    if cur_tokens <= allowed_budget:
        return preserved_header + modified_tail

    # Second pass: drop oldest message pairs (AIMessage with tool_calls + matching ToolMessages)
    while modified_tail and cur_tokens > allowed_budget:
        # Remove the oldest message from tail
        popped = modified_tail.pop(0)
        cur_tokens -= count_message_tokens(popped)

        # If we popped an AIMessage that had tool_calls, drop its immediately following ToolMessages too
        if isinstance(popped, AIMessage) and popped.tool_calls:
            expected_ids = {tc["id"] for tc in popped.tool_calls if "id" in tc}
            while modified_tail and isinstance(modified_tail[0], ToolMessage) and modified_tail[0].tool_call_id in expected_ids:
                tool_msg = modified_tail.pop(0)
                cur_tokens -= count_message_tokens(tool_msg)

    # Ensure the first message in the tail is not an orphaned ToolMessage
    while modified_tail and isinstance(modified_tail[0], ToolMessage):
        dropped = modified_tail.pop(0)
        cur_tokens -= count_message_tokens(dropped)

    return preserved_header + modified_tail


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


@tool
def append_documentation_section(
    section_title: str,
    markdown_content: str,
    level: int = 2
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
    # Strip any leading numbers, "Section X:", or Roman numerals the model may have generated
    clean_title = re.sub(r'^(?:Section\s*)?(?:\d+[\.\-_:]\s*)+', '', section_title.strip(), flags=re.IGNORECASE).strip()
    clean_title = re.sub(r'^(?:Section\s*)?[IVXLCDM]+[\.\-_:]\s*', '', clean_title, flags=re.IGNORECASE).strip()
    if not clean_title:
        clean_title = section_title.strip()

    # Automatically generate strict, monotonic, sequential numbering
    if level <= 2:
        _MAIN_SECTION_COUNTER += 1
        _SUB_SECTION_COUNTER = 0
        numbered_title = f"{_MAIN_SECTION_COUNTER}. {clean_title}"
    else:
        _SUB_SECTION_COUNTER += 1
        numbered_title = f"{_MAIN_SECTION_COUNTER}.{_SUB_SECTION_COUNTER} {clean_title}"

    prefix = "#" * max(1, min(level, 5))
    formatted_chunk = f"{prefix} {numbered_title}\n\n{markdown_content.strip()}\n\n---\n\n"

    try:
        with open(_DOC_OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(formatted_chunk)
            f.flush()

        _DOCUMENTED_SECTIONS.append({
            "title": numbered_title,
            "raw_title": clean_title,
            "level": level,
            "timestamp": time.time(),
            "preview": markdown_content[:120].replace("\n", " ")
        })

        console.print(f"  [bold green]📝 Appended Section (Level {level}):[/bold green] [cyan]{escape(numbered_title)}[/cyan]")
        return f"Successfully appended section '{numbered_title}' ({len(markdown_content)} chars) to documentation file. Total sections: {len(_DOCUMENTED_SECTIONS)}."
    except Exception as e:
        return f"Error writing to documentation file: {e}"


@tool
def read_current_documentation_toc() -> str:
    """Read the current Table of Contents of all sections that have already been written to disk.
    Use this to see what has already been documented and avoid duplicate sections.
    """
    global _DOCUMENTED_SECTIONS
    if not _DOCUMENTED_SECTIONS:
        return "No documentation sections have been written yet."

    lines = ["Current Documented Sections:"]
    for i, s in enumerate(_DOCUMENTED_SECTIONS, 1):
        indent = "  " * (s.get("level", 2) - 1)
        lines.append(f"{indent}- {s['title']}")
    return "\n".join(lines)


DOCUMENTER_TOOLS = [
    clangd_query,
    ripgrep_search,
    read_project_file,
    list_project_structure,
    append_documentation_section,
    read_current_documentation_toc
]


# ============================================================================
# Documenter System Prompt
# ============================================================================

DOCUMENTER_SYSTEM_PROMPT = """You are an expert Principal C++ Software Architect and Technical Writer.
Your task is to generate clear, comprehensive, functional, publication-grade Markdown documentation for the given C++ codebase.

════════════════════════════════════════════════════════════════════════════════
CORE INSTRUCTIONS & STANDARDS:
1. 📂 INCREMENTAL OUTPUT & CLEAN TITLES:
   - Call `append_documentation_section` as soon as you finish investigating each module or major workflow.
   - DO NOT provide section numbers in `section_title` (e.g. do NOT write 'Section 2' or '3. ...').
     Provide ONLY the descriptive semantic title (e.g. 'Order Processing Engine & Payment Workflow').
     Section numbers are automatically tracked and prefixed for you in sequence.
2. ⚙️ FOCUS ON FUNCTIONAL EXPLANATION & CODE INNER LOGIC:
   - Do NOT just produce a dry list of class names and field types.
   - Deeply explain WHAT the code does functionally: business logic, operational behavior, workflows, and state transitions.
   - EXPLAIN THE CRITICAL PARTS OF THE CODE:
     * Break down key algorithms and methods step-by-step.
     * Explain input validations, data transformations, error handling, return codes, and side effects.
     * Describe edge cases, failure recoveries, and performance considerations.
     * Explain dynamic interactions: trace how caller functions pass data into methods and what downstream effects occur.
3. 🎯 TECHNICAL ACCURACY & CONCRETENESS:
   - Always cite exact source files, classes, methods, and line numbers (`include/order_repository.h`).
   - Use `clangd_query` (`show`, `interface`, `hierarchy`, `signature`) to inspect actual implementations and method bodies.
   - Use `ripgrep_search` to verify concurrency primitives (`mutex`, `shared_mutex`, `atomic`) and resource ownership (`std::unique_ptr`, `std::shared_ptr`, RAII).
4. 📊 DIAGRAMS & FLOWCHARTS (MERMAID ONLY):
   - Whenever illustrating architecture, class relationships, state transitions, or execution flows, you MUST use Mermaid diagrams ONLY inside fenced code blocks (` ```mermaid ... ``` `).
   - Do NOT use ASCII art, plain text boxes, or pseudo-code drawings for diagrams.
   - Supported Mermaid types:
     * `flowchart TD` or `flowchart LR` for functional data/control pipelines.
     * `sequenceDiagram` for method call sequences and dynamic component interactions.
     * `classDiagram` for class inheritance and interface relationships.
   - Ensure valid Mermaid syntax: quote node labels containing special characters (parentheses, braces, brackets), e.g. `id["OrderRepository (Thread-Safe)"]`.
5. 🧠 CONTEXT EFFICIENCY (32k LIMIT):
   - Keep tool queries targeted and focused.
   - Append sections incrementally to keep the context window compact and clean.
════════════════════════════════════════════════════════════════════════════════
"""


# ============================================================================
# Multi-Node Documenter Graph
# ============================================================================

class DocumenterState(TypedDict):
    project_dir: str
    output_file: str
    target_dirs: Optional[List[str]]
    ignore_dirs: Optional[List[str]]
    all_files: List[str]
    modules: List[str]
    module_files_map: Dict[str, List[str]]
    current_module_index: int
    sections_count: int
    max_context_tokens: int


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

    module_map: Dict[str, List[str]] = {}
    for f in all_files:
        parent = str(Path(f).parent)
        if parent not in module_map:
            module_map[parent] = []
        module_map[parent].append(f)

    sorted_modules = sorted(list(module_map.keys()))

    console.print(f"[green]Discovered {len(all_files)} total C++ files across {len(sorted_modules)} directories/modules.[/green]")
    for mod in sorted_modules:
        console.print(f"  📁 [bold]{escape(mod)}/[/bold] ({len(module_map[mod])} files)")

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
        console.print(f"\n[bold cyan]🎯 Starting Point Detected:[/bold cyan] [green]{entry_point_files[0]}[/green] (Module: [yellow]{entry_point_module}/[/yellow])")
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
                console.print(f"  🎯 [bold cyan]{escape(mod)}/[/bold cyan] ({len(module_map[mod])} files)")
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
        tag = " [bold magenta]★ Entry Point[/bold magenta]" if (entry_point_module and mod == entry_point_module) else ""
        console.print(f"  {rank}. [bold cyan]{escape(mod)}/[/bold cyan] ({len(module_map[mod])} files){tag}")

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
        marker = "🎯 *(Documented)*" if mod in modules_to_document else "📦 *(Reference)*"
        sec1_content += f"- **`{mod}/`** ({len(module_map[mod])} files) {marker}:\n"
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
        "current_module_index": 0,
        "sections_count": 1
    }


def build_module_documenter_runner(llm, max_context_tokens: int = 32000):
    """
    Build focused LangGraph sub-agent for documenting a specific directory module,
    strictly bounded by the 32k context size limit.
    """
    llm_with_tools = llm.bind_tools(DOCUMENTER_TOOLS)

    def agent_step(state: MessagesState) -> Dict[str, Any]:
        # Enforce 32k context token limit
        trimmed_messages = trim_messages_to_budget(state["messages"], max_tokens=max_context_tokens)

        max_retries = 5
        base_delay = 6
        for attempt in range(1, max_retries + 1):
            try:
                response = llm_with_tools.invoke(trimmed_messages)
                return {"messages": [response]}
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

    tool_node = ToolNode(DOCUMENTER_TOOLS)
    wf = StateGraph(MessagesState)
    wf.add_node("agent", agent_step)
    wf.add_node("tools", tool_node)
    wf.add_edge(START, "agent")
    wf.add_conditional_edges("agent", tools_condition, ["tools", END])
    wf.add_edge("tools", "agent")
    return wf.compile()


def document_module_node_factory(llm, max_context_tokens: int = 32000, module_max_steps: int = 50):
    """
    Create document_module node with 32k context limitation, step control, and functional focus.
    """
    sub_agent = build_module_documenter_runner(llm, max_context_tokens=max_context_tokens)

    def document_module_node(state: DocumenterState) -> Dict[str, Any]:
        idx = state["current_module_index"]
        modules = state["modules"]
        current_module = modules[idx]
        files = state["module_files_map"].get(current_module, [])

        console.print(f"\n[bold yellow]═══ Phase 2: Documenting Module ({idx + 1}/{len(modules)}): [cyan]{escape(current_module)}/[/cyan] ({len(files)} files) ═══[/bold yellow]")

        prompt = (
            f"You are writing in-depth functional C++ documentation for module '{current_module}'.\n"
            f"Files in this module:\n" + "\n".join(f"- {f}" for f in files) + "\n\n"
            f"TASK & FOCUS REQUIREMENTS:\n"
            f"1. Investigate the symbols and implementations in these files using 'clangd_query' ('show', 'interface', 'signature') and 'read_project_file'.\n"
            f"2. FOCUS ON FUNCTIONAL BEHAVIOR: Explain what this module actually does in practice, how its algorithms work, and what role it plays in the overall system.\n"
            f"3. EXPLAIN CODE PARTS & LOGIC: Detail key methods, input parameters, transformations, concurrency locks (mutex/shared_mutex), return types, and error handling.\n"
            f"4. Add a Mermaid sequence or flowchart diagram (strictly Mermaid only, never ASCII art) illustrating the runtime execution flow or class interactions.\n"
            f"5. Call 'append_documentation_section' to save the section to disk.\n"
            f"   IMPORTANT: For 'section_title', provide only a descriptive title (e.g. '{current_module.replace('/', ' ').title()} - Functional Architecture & Implementation'). "
            f"   DO NOT include section numbers; numbering is managed automatically in sequence.\n"
            f"6. Conclude once 'append_documentation_section' has been called."
        )

        sub_state: MessagesState = {
            "messages": [
                SystemMessage(content=DOCUMENTER_SYSTEM_PROMPT),
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
                                console.print(f"  [magenta]▶ Tool:[/magenta] [cyan]{escape(tc['name'])}[/cyan]({escape(json.dumps(tc['args']))})")
                    elif node_name == "tools":
                        for msg in node_update["messages"]:
                            raw_text = extract_text(msg.content)
                            preview = raw_text[:120].replace("\n", " ")
                            if len(raw_text) > 120:
                                preview += "..."
                            console.print(f"[dim]    ↳ Result: {escape(preview)}[/dim]")
        except Exception as e:
            console.print(f"[dim yellow]  (Module '{escape(current_module)}' documentation completed or reached step budget: {e})[/dim yellow]")

        pct = ((idx + 1) / len(modules)) * 100
        console.print(f"[green]✔ Finished Module '{escape(current_module)}/' ({idx + 1}/{len(modules)} modules - {pct:.1f}% complete)[/green]")

        return {
            "current_module_index": idx + 1,
            "sections_count": state.get("sections_count", 0) + 1
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

    console.print(f"[bold green]✔ Documentation complete![/bold green] Total sections documented: {len(_DOCUMENTED_SECTIONS)}")
    console.print(f"[bold green]Saved to:[/bold green] [cyan]{out_file}[/cyan]\n")
    return {}


def build_codebase_documenter_graph(llm, max_context_tokens: int = 32000, module_max_steps: int = 50):
    """
    Build the deterministic multi-node documentation orchestrator.
    """
    wf = StateGraph(DocumenterState)

    wf.add_node("discover_and_plan", doc_discover_and_plan_node)
    wf.add_node("document_module", document_module_node_factory(llm, max_context_tokens=max_context_tokens, module_max_steps=module_max_steps))
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

    console.print(Panel(
        f"[bold cyan]Autonomous C++ Codebase Documentation Agent[/bold cyan]\n"
        f"Project Path    : [yellow]{proj_path}[/yellow]\n"
        f"Target Scope    : {target_display}\n"
        f"Ignored Folders : {ignore_display}\n"
        f"Output Document : [green]{out_file}[/green]\n"
        f"Provider        : [green]{provider}[/green]\n"
        f"Model           : [green]{model_name}[/green]\n"
        f"Context Limit   : [magenta]{max_context_tokens:,} tokens (32k window guard)[/magenta]\n"
        f"Steps / Module  : [yellow]{module_max_steps}[/yellow]\n"
        f"Tools Active    : [blue]clangd-query, ripgrep (rg), read_project_file, append_documentation_section[/blue]",
        title="Agent Configuration",
        border_style="cyan"
    ))

    llm = get_llm(provider=provider, model_name=model_name, ollama_host=ollama_host)
    app = build_codebase_documenter_graph(
        llm=llm,
        max_context_tokens=max_context_tokens,
        module_max_steps=module_max_steps
    )

    initial_state: DocumenterState = {
        "project_dir": str(proj_path),
        "output_file": str(out_file),
        "target_dirs": target_dirs,
        "ignore_dirs": ignore_dirs,
        "all_files": [],
        "modules": [],
        "module_files_map": {},
        "current_module_index": 0,
        "sections_count": 0,
        "max_context_tokens": max_context_tokens
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
        "--max-steps",
        type=int,
        default=50,
        help="Maximum tool steps per module exploration (default: 50)"
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
        module_max_steps=args.max_steps,
        target_dirs=targets,
        ignore_dirs=ignores
    )
