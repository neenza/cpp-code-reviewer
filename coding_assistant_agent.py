#!/usr/bin/env python3
"""
General C++ Coding Assistant Agent.
Interactive and goal-driven software engineering assistant equipped with AST queries (clangd-query),
fast regex search (ripgrep), file reading, full file creation (write_project_file),
precise chunk patching (edit_project_file), and shell command execution (execute_shell_command).
Features interactive user permission confirmations for modifications and automatic rolling context summarization.
"""

import os
import sys
import json
import time
import argparse
from typing import Optional, List, Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.markup import escape

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from cpp_agent_tools import (
    clangd_query,
    ripgrep_search,
    read_project_file,
    list_project_structure,
    write_project_file,
    edit_project_file,
    execute_shell_command,
    CODING_ASSISTANT_TOOLS,
    set_active_project_dir,
    get_active_project_dir,
    ensure_compile_commands,
    get_llm,
    extract_text,
    extract_message_text,
    count_tokens,
    count_message_tokens,
    partition_messages_safely,
    set_require_permission,
    get_require_permission
)

load_dotenv()
console = Console()

# ============================================================================
# System Prompt & Persona
# ============================================================================

CODING_ASSISTANT_SYSTEM_PROMPT = """You are an expert C++ Software Engineer and Autonomous Coding Assistant.
Your mission is to help the user investigate, design, implement, modify, debug, refactor, and test code in this C++ repository.

════════════════════════════════════════════════════════════════════════════════
CORE ENGINEERING PRINCIPLES:
1. INVESTIGATE BEFORE MODIFYING:
   - Always inspect relevant code before proposing or making changes.
   - Use 'clangd_query(command="interface", symbol_or_query="<ClassName>")' to inspect class declarations.
   - Use 'clangd_query(command="show", symbol_or_query="<ClassName::MethodName>")' to inspect method implementations.
   - Use 'clangd_query(command="usages", symbol_or_query="<SymbolName>")' to inspect callers and dependencies.
   - Use 'ripgrep_search' to locate keywords, configs, macros, or error strings across files.
   - Never guess function signatures, member variable names, or include paths.

2. SURGICAL & MINIMAL FILE EDITS:
   - Prefer 'edit_project_file' for targeted chunk replacements over rewriting entire files.
   - Ensure 'target_content' matches the existing file contents exactly, including all whitespace and indentation.
   - Use 'write_project_file' when creating whole new source/header files or build configs.

3. VERIFICATION & BUILD VALIDATION:
   - When appropriate, verify that modified code builds or tests pass using 'execute_shell_command'.
   - If a build fails or tests produce errors, inspect the error messages and iteratively fix the code.

4. USER PERMISSION PROTOCOL:
   - All file modifications ('write_project_file', 'edit_project_file') and shell executions ('execute_shell_command')
     automatically require user confirmation before execution.
   - If the user denies permission for an action, respect their decision, understand the constraint, and either
     propose an alternative approach or ask clarifying questions.

5. CONTEXT PRESERVATION:
   - Do NOT read large files (>80 lines) completely; use targeted line ranges or AST tools.
   - When explaining your changes, cite exact file paths and line numbers.
════════════════════════════════════════════════════════════════════════════════
"""

# ============================================================================
# Context Management with Rolling Summarization
# ============================================================================

def print_context_banner(messages: List[BaseMessage], max_tokens: int, stage: str = "Assistant Step") -> int:
    """Display current context token usage and utilization percentage."""
    total_tokens = sum(count_message_tokens(m) for m in messages)
    pct = (total_tokens / max(1, max_tokens)) * 100
    color = "green" if pct < 45 else ("yellow" if pct < 75 else "bold red")
    console.print(f"  [{color}][Context: {total_tokens:,} / {max_tokens:,} tokens ({pct:.1f}%)] | {stage}[/{color}]")
    return total_tokens


def manage_context_with_summarization(
    messages: List[BaseMessage],
    llm,
    max_tokens: int = 32000,
    reserve_tokens: int = 3000,
    summarize_threshold: Optional[int] = None
) -> List[BaseMessage]:
    """
    Proactively compresses conversation history when total tokens reach summarize_threshold.
    Preserves recent active turns and in-flight tool messages without truncation.
    """
    if summarize_threshold is None:
        summarize_threshold = min(12000, int(max_tokens * 0.45))
    effective_limit = min(summarize_threshold, max_tokens - reserve_tokens)

    cur_tokens = sum(count_message_tokens(m) for m in messages)
    if cur_tokens < effective_limit:
        return messages

    console.print(
        f"\n  [bold yellow][Notice] Context reached {cur_tokens:,} tokens "
        f"(exceeding threshold {effective_limit:,}). Active Rolling Summarization initiated...[/bold yellow]"
    )

    # Preserve initial system prompt and root user instruction
    preserved_header = []
    if messages and isinstance(messages[0], SystemMessage):
        preserved_header.append(messages[0])
        tail_start = 1
    else:
        preserved_header.append(SystemMessage(content=CODING_ASSISTANT_SYSTEM_PROMPT))
        tail_start = 0

    if len(messages) > tail_start and isinstance(messages[tail_start], HumanMessage):
        preserved_header.append(messages[tail_start])
        conversation_tail = list(messages[tail_start + 1:])
    else:
        conversation_tail = list(messages[tail_start:])

    older, recent = partition_messages_safely(conversation_tail, target_recent_count=4)
    if not older:
        return messages

    history_lines = []
    for m in older:
        role = "System" if isinstance(m, SystemMessage) else (
            "User" if isinstance(m, HumanMessage) else (
                "Assistant" if isinstance(m, AIMessage) else "Tool"
            )
        )
        content_txt = extract_text(m.content)
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            calls_summary = ", ".join(tc.get("name", "") for tc in m.tool_calls)
            history_lines.append(f"Assistant invoked tools: {calls_summary}")
        if content_txt:
            lines = content_txt.splitlines()
            preview = "\n".join(lines[:30])
            if len(lines) > 30:
                preview += f"\n... [Truncated {len(lines) - 30} lines]"
            history_lines.append(f"[{role}]: {preview}")

    history_text = "\n\n".join(history_lines)

    summary_prompt = (
        "You are condensing conversation and tool execution history for a C++ coding assistant.\n"
        "Summarize the technical exploration, file changes, and decisions made so far into a concise Markdown brief.\n\n"
        "Include:\n"
        "1. **User Goal & Tasks**: What the user requested.\n"
        "2. **Files Explored & Modified**: Files examined, written, or patched.\n"
        "3. **Key Discoveries**: Important AST classes, functions, bugs, or architecture noted.\n"
        "4. **Current Status**: What has been completed and what work remains.\n\n"
        f"Conversation history to summarize:\n{history_text}"
    )

    try:
        summary_resp = llm.invoke([
            SystemMessage(content="You are a concise technical summarizer preserving coding context."),
            HumanMessage(content=summary_prompt)
        ])
        summary_body = extract_text(summary_resp.content).strip()
    except Exception as e:
        console.print(f"[yellow]Summarization failed ({e}). Using fallback brief.[/yellow]")
        summary_body = f"Prior conversation spanned {len(older)} turns focusing on codebase inspection and editing."

    summary_msg = SystemMessage(
        content=(
            f"════════════════════════════════════════════════════════════════════════════════\n"
            f"ACTIVE ROLLING CONTEXT SUMMARY (Prior Turns Condensed):\n"
            f"{summary_body}\n"
            f"════════════════════════════════════════════════════════════════════════════════"
        )
    )

    condensed_messages = preserved_header + [summary_msg] + recent
    new_tokens = sum(count_message_tokens(m) for m in condensed_messages)
    saved_tokens = cur_tokens - new_tokens
    pct_saved = (saved_tokens / max(1, cur_tokens)) * 100

    console.print(
        f"  [bold green][OK] Context condensed from {cur_tokens:,} down to {new_tokens:,} tokens "
        f"(-{pct_saved:.1f}%, saved {saved_tokens:,} tokens).[/bold green]\n"
    )

    return condensed_messages


# ============================================================================
# Graph Builder
# ============================================================================

def build_coding_assistant_graph(
    llm,
    tools: Optional[List[Any]] = None,
    max_context_tokens: int = 32000,
    summarize_threshold: Optional[int] = None
):
    """
    Build the interactive and one-shot LangGraph coding assistant workflow.
    """
    active_tools = tools if tools is not None else CODING_ASSISTANT_TOOLS
    llm_with_tools = llm.bind_tools(active_tools)

    def agent_step(state: MessagesState) -> Dict[str, Any]:
        raw_messages = state["messages"]

        # 1. Manage context limits with proactive rolling summarization
        compact_messages = manage_context_with_summarization(
            raw_messages,
            llm=llm,
            max_tokens=max_context_tokens,
            summarize_threshold=summarize_threshold
        )

        # 2. Display context usage banner
        print_context_banner(compact_messages, max_tokens=max_context_tokens, stage="Agent Step")

        # 3. Invoke LLM with rate limit retries
        max_retries = 5
        base_delay = 6
        for attempt in range(1, max_retries + 1):
            try:
                response = llm_with_tools.invoke(compact_messages)
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

        # Print tool calls if present
        if getattr(response, "tool_calls", None):
            for tc in response.tool_calls:
                call_args = json.dumps(tc.get("args", {}))
                if len(call_args) > 120:
                    call_args = call_args[:120] + "..."
                console.print(f"  [magenta]Tool Request:[/magenta] [cyan]{escape(tc.get('name', ''))}[/cyan]({escape(call_args)})")
        else:
            txt = extract_message_text(response)
            if txt:
                preview = txt[:150].replace("\n", " ")
                if len(txt) > 150:
                    preview += "..."
                console.print(f"  [dim]Assistant Response: {escape(preview)}[/dim]")

        # If context was summarized, replace messages; otherwise append response
        if len(compact_messages) != len(raw_messages):
            return {"messages": compact_messages + [response]}
        return {"messages": [response]}

    def route_assistant_step(state: MessagesState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return END
        last_msg = messages[-1]
        if getattr(last_msg, "tool_calls", None):
            return "tools"
        return END

    tool_node = ToolNode(active_tools)

    wf = StateGraph(MessagesState)
    wf.add_node("agent", agent_step)
    wf.add_node("tools", tool_node)

    wf.add_edge(START, "agent")
    wf.add_conditional_edges("agent", route_assistant_step, {
        "tools": "tools",
        END: END
    })
    wf.add_edge("tools", "agent")

    return wf.compile()


# ============================================================================
# Assistant Runners: Interactive REPL & One-Shot Mode
# ============================================================================

def run_interactive_assistant(
    project_dir: str,
    provider: str = "gemini",
    model_name: Optional[str] = None,
    ollama_host: str = "http://localhost:11434",
    max_context_tokens: int = 32000,
    summarize_threshold: int = 12000,
    max_steps: int = 60,
    auto_approve: bool = False
) -> None:
    """Start an interactive coding assistant REPL session."""
    proj_path = Path(project_dir).resolve()
    if not proj_path.exists() or not proj_path.is_dir():
        console.print(f"[bold red]Error: Project directory '{project_dir}' does not exist.[/bold red]")
        sys.exit(1)

    set_active_project_dir(proj_path)
    ensure_compile_commands(proj_path)
    set_require_permission(not auto_approve)

    if not model_name:
        model_name = "llama3.1:8b" if provider.lower() == "ollama" else "gemini-2.5-flash"

    console.print(Panel(
        f"[bold cyan]C++ General Coding Assistant[/bold cyan]\n"
        f"Project Path    : [yellow]{proj_path}[/yellow]\n"
        f"Provider        : [green]{provider}[/green]\n"
        f"Model           : [green]{model_name}[/green]\n"
        f"Context Limit   : [magenta]{max_context_tokens:,} tokens[/magenta]\n"
        f"Summarize Limit : [magenta]{summarize_threshold:,} tokens[/magenta]\n"
        f"Permissions     : [yellow]{'Interactive Prompts' if not auto_approve else 'Auto-Approved (--yes)'}[/yellow]\n"
        f"Tools Active    : [blue]clangd_query, ripgrep, read/write/edit_project_file, execute_shell_command[/blue]\n\n"
        f"[dim]Type your requests or questions. Type 'exit' or 'quit' to end session.[/dim]",
        title="Session Initialized",
        border_style="cyan"
    ))

    llm = get_llm(
        provider=provider,
        model_name=model_name,
        ollama_host=ollama_host,
        max_context_tokens=max_context_tokens
    )

    app = build_coding_assistant_graph(
        llm=llm,
        max_context_tokens=max_context_tokens,
        summarize_threshold=summarize_threshold
    )

    conversation_messages: List[BaseMessage] = [
        SystemMessage(content=CODING_ASSISTANT_SYSTEM_PROMPT)
    ]

    while True:
        try:
            console.print("\n[bold green]You >[/bold green] ", end="")
            user_input = input().strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session ended.[/yellow]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[green]Goodbye![/green]")
            break
        if user_input.lower() == "clear":
            conversation_messages = [SystemMessage(content=CODING_ASSISTANT_SYSTEM_PROMPT)]
            console.print("[dim]Conversation history cleared.[/dim]")
            continue

        conversation_messages.append(HumanMessage(content=user_input))

        try:
            result = app.invoke(
                {"messages": conversation_messages},
                {"recursion_limit": max_steps}
            )
            conversation_messages = result.get("messages", conversation_messages)
            last_msg = conversation_messages[-1]
            content_text = extract_text(last_msg.content)
            if content_text:
                console.print(Panel(Markdown(content_text), title="Assistant", border_style="green"))
        except Exception as e:
            console.print(f"[bold red]Execution error:[/bold red] {e}")


def run_oneshot_assistant(
    project_dir: str,
    prompt: str,
    provider: str = "gemini",
    model_name: Optional[str] = None,
    ollama_host: str = "http://localhost:11434",
    max_context_tokens: int = 32000,
    summarize_threshold: int = 12000,
    max_steps: int = 60,
    auto_approve: bool = False
) -> str:
    """Execute a single-shot goal or instruction autonomously."""
    proj_path = Path(project_dir).resolve()
    if not proj_path.exists() or not proj_path.is_dir():
        console.print(f"[bold red]Error: Project directory '{project_dir}' does not exist.[/bold red]")
        sys.exit(1)

    set_active_project_dir(proj_path)
    ensure_compile_commands(proj_path)
    set_require_permission(not auto_approve)

    if not model_name:
        model_name = "llama3.1:8b" if provider.lower() == "ollama" else "gemini-2.5-flash"

    console.print(Panel(
        f"[bold cyan]C++ General Coding Assistant (Autonomous Task)[/bold cyan]\n"
        f"Project Path    : [yellow]{proj_path}[/yellow]\n"
        f"Directive       : [white]{prompt}[/white]\n"
        f"Provider        : [green]{provider}[/green]\n"
        f"Model           : [green]{model_name}[/green]\n"
        f"Permissions     : [yellow]{'Interactive Prompts' if not auto_approve else 'Auto-Approved (--yes)'}[/yellow]",
        title="Task Execution",
        border_style="cyan"
    ))

    llm = get_llm(
        provider=provider,
        model_name=model_name,
        ollama_host=ollama_host,
        max_context_tokens=max_context_tokens
    )

    app = build_coding_assistant_graph(
        llm=llm,
        max_context_tokens=max_context_tokens,
        summarize_threshold=summarize_threshold
    )

    initial_messages: List[BaseMessage] = [
        SystemMessage(content=CODING_ASSISTANT_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    result = app.invoke(
        {"messages": initial_messages},
        {"recursion_limit": max_steps}
    )

    last_msg = result.get("messages", [])[-1]
    final_text = extract_text(last_msg.content)
    console.print("\n" + "=" * 80)
    console.print(Panel(Markdown(final_text), title="Task Completed", border_style="bold green"))
    return final_text


# ============================================================================
# CLI Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="General C++ Coding Assistant with AST intelligence, file write/patch tools, and rolling context summarization."
    )
    parser.add_argument(
        "project_dir",
        type=str,
        nargs="?",
        default=".",
        help="Path to the C++ project root directory (default: current directory)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="One-shot prompt / task directive. If omitted, starts interactive REPL."
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "ollama"],
        default="gemini",
        help="LLM provider: 'gemini' or 'ollama' (default: gemini)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: 'gemini-2.5-flash' for gemini, 'llama3.1:8b' for ollama)"
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
        help="Maximum context token budget (default: 32000)"
    )
    parser.add_argument(
        "--context-summarize-threshold",
        type=int,
        default=12000,
        help="Token threshold to proactively trigger rolling context summarization (default: 12000)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60,
        help="Maximum recursion steps per user instruction (default: 60)"
    )
    parser.add_argument(
        "--auto-approve",
        "--yes",
        "-y",
        action="store_true",
        help="Automatically grant permission for file writes, patches, and shell commands without prompting."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.prompt:
        run_oneshot_assistant(
            project_dir=args.project_dir,
            prompt=args.prompt,
            provider=args.provider,
            model_name=args.model,
            ollama_host=args.ollama_host,
            max_context_tokens=args.max_context_tokens,
            summarize_threshold=args.context_summarize_threshold,
            max_steps=args.max_steps,
            auto_approve=args.auto_approve
        )
    else:
        run_interactive_assistant(
            project_dir=args.project_dir,
            provider=args.provider,
            model_name=args.model,
            ollama_host=args.ollama_host,
            max_context_tokens=args.max_context_tokens,
            summarize_threshold=args.context_summarize_threshold,
            max_steps=args.max_steps,
            auto_approve=args.auto_approve
        )
