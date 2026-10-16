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
from typing import Optional, List, Dict, Any, TypedDict, Annotated
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.markup import escape

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
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
    invoke_llm_with_retries,
    set_current_context_stats,
    extract_text,
    extract_message_text,
    count_tokens,
    count_message_tokens,
    partition_messages_safely,
    set_require_permission,
    get_require_permission
)
from mcp_tool_adapter import build_mcp_tools

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
   - For C/C++ files: use 'clangd_query(command="interface", symbol_or_query="ClassName")' to inspect class declarations,
     'clangd_query(command="show", symbol_or_query="ClassName::MethodName")' for method implementations,
     and 'clangd_query(command="usages", symbol_or_query="SymbolName")' for callers and dependencies.
   - For Python or non-C++ files: use 'ripgrep_search' to locate function/class definitions, variables, or error strings with line numbers first.
   - Never guess function signatures, member variable names, or include paths.

2. SURGICAL & MINIMAL FILE EDITS:
   - Prefer 'edit_project_file' for targeted chunk replacements over rewriting entire files.
   - Ensure 'target_content' matches the existing file contents exactly, including all whitespace and indentation.
   - Use 'write_project_file' when creating whole new source/header files or build configs.

3. VERIFICATION & BUILD VALIDATION:
   - When appropriate, verify that modified code builds or tests pass using 'execute_shell_command'.
   - If a build fails or tests produce errors, inspect the error messages and iteratively fix the code.

4. USER PERMISSION & FEEDBACK PROTOCOL:
   - All file modifications ('write_project_file', 'edit_project_file') and shell executions ('execute_shell_command')
     automatically require user confirmation before execution.
   - If the user denies permission or provides an explicit rejection reason/feedback, treat their feedback as a hard constraint.
     Immediately adjust your approach, patch, or command to address their feedback and proceed towards the user's objective.

5. CONTEXT PRESERVATION:
   - When reading files with 'read_project_file', specify targeted line ranges (up to 1,000 lines) or AST/search tools.
   - Full file reads are supported; for large files (>120 lines), the tool prompts with token estimates on first request.
   - When explaining your changes, cite exact file paths and line numbers.

6. TOOL CALLING SYNTAX & PROTOCOL:
   - Ensure all tool call arguments are well-formed strings.
   - When calling tools that use XML formatting (such as local Ollama models), strictly ensure every opened '<parameter>' tag is closed with its matching '</parameter>' tag before closing '</function>'.
   - Avoid raw, unescaped markup tags or unmatched angle brackets inside parameter arguments that could disrupt XML parsing.
════════════════════════════════════════════════════════════════════════════════
"""


def load_system_prompt(system_prompt_file: Optional[str] = None) -> str:
    """Load system prompt from file if provided, combining base engineering guidelines with custom instructions."""
    if not system_prompt_file:
        return CODING_ASSISTANT_SYSTEM_PROMPT

    path = Path(system_prompt_file).resolve()
    if not path.exists():
        console.print(f"[bold red]Error: System prompt file '{system_prompt_file}' does not exist.[/bold red]")
        sys.exit(1)
    if path.is_dir():
        console.print(f"[bold red]Error: System prompt path '{system_prompt_file}' is a directory, not a file.[/bold red]")
        sys.exit(1)

    try:
        custom_content = path.read_text(encoding="utf-8").strip()
    except Exception as e:
        console.print(f"[bold red]Error reading system prompt file '{system_prompt_file}': {e}[/bold red]")
        sys.exit(1)

    if not custom_content:
        console.print(f"[yellow]Warning: System prompt file '{system_prompt_file}' is empty. Using default system prompt.[/yellow]")
        return CODING_ASSISTANT_SYSTEM_PROMPT

    # If the custom prompt already contains full assistant persona/principles, use it directly;
    # otherwise append to base tool principles to ensure tool-calling competency is preserved.
    if "CORE ENGINEERING PRINCIPLES" in custom_content or "clangd_query" in custom_content:
        return custom_content

    return (
        f"{CODING_ASSISTANT_SYSTEM_PROMPT}\n"
        f"════════════════════════════════════════════════════════════════════════════════\n"
        f"CUSTOM SYSTEM INSTRUCTIONS & CONSTRAINTS (Loaded from {path.name}):\n"
        f"The following user instructions are authoritative, must be strictly followed across all turns,\n"
        f"and override any conflicting default behaviors:\n\n"
        f"{custom_content}\n"
        f"════════════════════════════════════════════════════════════════════════════════\n"
    )


# ============================================================================
# Context Management with Rolling Summarization
# ============================================================================

def print_context_banner(messages: List[BaseMessage], max_tokens: int, stage: str = "Assistant Step") -> int:
    """Display current context token usage and utilization percentage."""
    total_tokens = sum(count_message_tokens(m) for m in messages)
    set_current_context_stats(total_tokens, max_tokens)
    pct = (total_tokens / max(1, max_tokens)) * 100
    color = "green" if pct < 45 else ("yellow" if pct < 75 else "bold red")
    console.print(f"  [{color}][Context: {total_tokens:,} / {max_tokens:,} tokens ({pct:.1f}%)] | {stage}[/{color}]")
    return total_tokens


def manage_context_with_summarization(
    messages: List[BaseMessage],
    llm,
    max_tokens: int = 32000,
    reserve_tokens: int = 3000,
    summarize_threshold: Optional[int] = None,
    default_system_prompt: Optional[str] = None
) -> List[BaseMessage]:
    """
    Proactively compresses conversation history when total tokens reach summarize_threshold.
    Preserves recent active turns and in-flight tool messages without truncation.
    Permanently preserves the root system prompt (including custom instructions files) intact.
    """
    effective_reserve = min(reserve_tokens, int(max_tokens * 0.15))
    if summarize_threshold is None:
        summarize_threshold = min(12000, int(max_tokens * 0.45))
    effective_limit = min(summarize_threshold, max(1, max_tokens - effective_reserve))

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
        fallback_prompt = default_system_prompt or CODING_ASSISTANT_SYSTEM_PROMPT
        preserved_header.append(SystemMessage(content=fallback_prompt))
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
# State & Graph Builder
# ============================================================================

def message_reducer(existing: List[BaseMessage], update: Any) -> List[BaseMessage]:
    """Custom reducer supporting list concatenation and explicit context override upon summarization."""
    if isinstance(update, tuple) and len(update) == 2 and update[0] == "override":
        return list(update[1])
    if isinstance(update, list):
        return list(existing) + list(update)
    if isinstance(update, BaseMessage):
        return list(existing) + [update]
    return list(existing)


class CodingAssistantState(TypedDict):
    messages: Annotated[List[BaseMessage], message_reducer]


def build_coding_assistant_graph(
    llm,
    tools: Optional[List[Any]] = None,
    max_context_tokens: int = 32000,
    summarize_threshold: Optional[int] = None,
    system_prompt: Optional[str] = None
):
    """
    Build the interactive and one-shot LangGraph coding assistant workflow.
    """
    active_tools = tools if tools is not None else CODING_ASSISTANT_TOOLS
    llm_with_tools = llm.bind_tools(active_tools)

    def agent_step(state: CodingAssistantState) -> Dict[str, Any]:
        raw_messages = state["messages"]

        # 1. Manage context limits with proactive rolling summarization
        compact_messages = manage_context_with_summarization(
            raw_messages,
            llm=llm,
            max_tokens=max_context_tokens,
            summarize_threshold=summarize_threshold,
            default_system_prompt=system_prompt
        )

        # 2. Display context usage banner
        print_context_banner(compact_messages, max_tokens=max_context_tokens, stage="Agent Step")

        # 3. Invoke LLM with rate limit & tool-syntax recovery retries
        response = invoke_llm_with_retries(
            llm=llm_with_tools,
            messages=compact_messages,
            max_retries=5,
            stage_name="Coding Assistant Step"
        )

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

        # If context was summarized, override state messages; otherwise append response
        if len(compact_messages) != len(raw_messages):
            return {"messages": ("override", compact_messages + [response])}
        return {"messages": [response]}

    def route_assistant_step(state: CodingAssistantState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return END
        last_msg = messages[-1]
        if getattr(last_msg, "tool_calls", None):
            return "tools"
        return END

    # Enforce sequential execution (max_concurrency=1) so multiple interactive file edits
    # execute one after another without console interleaving or file-overwrite race conditions.
    tool_node = ToolNode(active_tools).with_config(max_concurrency=1)

    wf = StateGraph(CodingAssistantState)
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
    auto_approve: bool = False,
    system_prompt_file: Optional[str] = None,
    mcp_server_url: Optional[str] = None,
    mcp_tools_info: Optional[str] = None,
    mcp_api_key: Optional[str] = None,
    mcp_headers: Optional[str] = None,
    mcp_config: Optional[str] = None
) -> None:
    """Start an interactive coding assistant REPL session."""
    proj_path = Path(project_dir).resolve()
    if not proj_path.exists() or not proj_path.is_dir():
        console.print(f"[bold red]Error: Project directory '{project_dir}' does not exist.[/bold red]")
        sys.exit(1)

    set_active_project_dir(proj_path)
    ensure_compile_commands(proj_path)
    set_require_permission(not auto_approve)

    system_prompt = load_system_prompt(system_prompt_file)

    if not model_name:
        model_name = "llama3.1:8b" if provider.lower() == "ollama" else "gemini-2.5-flash"

    prompt_source = f"File: {Path(system_prompt_file).name}" if system_prompt_file else "Default Engineering Principles"

    # Load user-configured MCP tools if provided
    mcp_tools: List[Any] = []
    if mcp_config or mcp_server_url:
        try:
            mcp_tools = build_mcp_tools(
                server_url=mcp_server_url,
                tools_info=mcp_tools_info,
                api_key=mcp_api_key,
                headers=mcp_headers,
                config_path=mcp_config
            )
            console.print(f"  [bold green][MCP][/bold green] Successfully connected. Loaded {len(mcp_tools)} MCP tool(s).")
        except Exception as e:
            console.print(f"  [bold red][MCP Error][/bold red] Failed to load MCP tools: {e}")

    tools_summary = "clangd_query, ripgrep, read/write/edit_project_file, execute_shell_command"
    if mcp_tools:
        tools_summary += f" + {len(mcp_tools)} MCP tool(s) ({', '.join(t.name for t in mcp_tools)})"

    console.print(Panel(
        f"[bold cyan]C++ General Coding Assistant[/bold cyan]\n"
        f"Project Path    : [yellow]{proj_path}[/yellow]\n"
        f"Provider        : [green]{provider}[/green]\n"
        f"Model           : [green]{model_name}[/green]\n"
        f"System Prompt   : [green]{prompt_source}[/green]\n"
        f"Context Limit   : [magenta]{max_context_tokens:,} tokens[/magenta]\n"
        f"Summarize Limit : [magenta]{summarize_threshold:,} tokens[/magenta]\n"
        f"Permissions     : [yellow]{'Interactive Prompts' if not auto_approve else 'Auto-Approved (--yes)'}[/yellow]\n"
        f"Tools Active    : [blue]{tools_summary}[/blue]\n\n"
        f"[dim]Type your requests or questions. Commands: /mcp list, /mcp add, /mcp clear, /help, exit.[/dim]",
        title="Session Initialized",
        border_style="cyan"
    ))

    llm = get_llm(
        provider=provider,
        model_name=model_name,
        ollama_host=ollama_host,
        max_context_tokens=max_context_tokens
    )

    def _build_effective_prompt(base_prompt: str, tools_list: List[Any]) -> str:
        if not tools_list:
            return base_prompt
        tool_names = ", ".join(t.name for t in tools_list)
        return (
            f"{base_prompt}\n"
            f"════════════════════════════════════════════════════════════════════════════════\n"
            f"USER-CONFIGURED MCP TOOLS:\n"
            f"You are also equipped with the following custom MCP tools: {tool_names}.\n"
            f"Use them whenever relevant to fulfill user directives.\n"
            f"════════════════════════════════════════════════════════════════════════════════\n"
        )

    effective_prompt = _build_effective_prompt(system_prompt, mcp_tools)
    all_active_tools = list(CODING_ASSISTANT_TOOLS) + list(mcp_tools)

    app = build_coding_assistant_graph(
        llm=llm,
        tools=all_active_tools,
        max_context_tokens=max_context_tokens,
        summarize_threshold=summarize_threshold,
        system_prompt=effective_prompt
    )

    conversation_messages: List[BaseMessage] = [
        SystemMessage(content=effective_prompt)
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
            conversation_messages = [SystemMessage(content=effective_prompt)]
            console.print("[dim]Conversation history cleared.[/dim]")
            continue

        if user_input.startswith("/mcp"):
            parts = user_input.split(maxsplit=2)
            subcmd = parts[1].lower() if len(parts) > 1 else "list"
            if subcmd == "list":
                if not mcp_tools:
                    console.print("  [yellow]No custom MCP tools are currently connected. Use '/mcp add' to connect an MCP server.[/yellow]")
                else:
                    console.print(f"\n  [bold cyan]Active MCP Tools ({len(mcp_tools)}):[/bold cyan]")
                    for t in mcp_tools:
                        console.print(f"  - [bold green]{escape(t.name)}[/bold green]: {escape(t.description)}")
                continue
            elif subcmd == "add":
                try:
                    console.print("\n  [bold cyan]── Connect Custom MCP Server ──[/bold cyan]")
                    url_in = Prompt.ask("  [bold green]MCP Server URL[/bold green]").strip()
                    if not url_in:
                        console.print("  [yellow]Connection cancelled (empty URL).[/yellow]")
                        continue
                    tools_in = Prompt.ask("  [bold green]Tools Info (path to .json file, dir, or 'auto')[/bold green]", default="auto").strip()
                    key_in = Prompt.ask("  [bold green]API Key / Token (optional, press Enter to skip)[/bold green]", default="").strip()
                    new_tools = build_mcp_tools(
                        server_url=url_in,
                        tools_info=tools_in,
                        api_key=key_in or None
                    )
                    mcp_tools.extend(new_tools)
                    effective_prompt = _build_effective_prompt(system_prompt, mcp_tools)
                    all_active_tools = list(CODING_ASSISTANT_TOOLS) + list(mcp_tools)
                    app = build_coding_assistant_graph(
                        llm=llm,
                        tools=all_active_tools,
                        max_context_tokens=max_context_tokens,
                        summarize_threshold=summarize_threshold,
                        system_prompt=effective_prompt
                    )
                    console.print(f"  [bold green]Successfully connected![/bold green] Added {len(new_tools)} MCP tool(s). (Total MCP tools: {len(mcp_tools)})\n")
                except Exception as e:
                    console.print(f"  [bold red]Failed to connect MCP server:[/bold red] {e}\n")
                continue
            elif subcmd in ("clear", "remove", "disconnect"):
                mcp_tools.clear()
                effective_prompt = system_prompt
                all_active_tools = list(CODING_ASSISTANT_TOOLS)
                app = build_coding_assistant_graph(
                    llm=llm,
                    tools=all_active_tools,
                    max_context_tokens=max_context_tokens,
                    summarize_threshold=summarize_threshold,
                    system_prompt=effective_prompt
                )
                console.print("  [green]Disconnected all MCP tools. Restored standard C++ tools.[/green]")
                continue
            else:
                console.print("  [dim]Usage: /mcp list | /mcp add | /mcp clear[/dim]")
                continue

        if user_input.lower() in ("/help", "help"):
            console.print(Panel(
                "[bold cyan]Available Commands:[/bold cyan]\n"
                "- [green]/mcp list[/green]     : Display connected MCP tools and descriptions\n"
                "- [green]/mcp add[/green]      : Interactively connect any custom MCP server\n"
                "- [green]/mcp clear[/green]    : Disconnect all active MCP tools\n"
                "- [green]clear[/green]         : Reset conversation context\n"
                "- [green]exit / quit[/green]   : End session",
                title="Help",
                border_style="cyan"
            ))
            continue

        conversation_messages.append(HumanMessage(content=user_input))

        try:
            result = app.invoke(
                {"messages": conversation_messages},
                {"recursion_limit": max_steps, "max_concurrency": 1}
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
    auto_approve: bool = False,
    system_prompt_file: Optional[str] = None,
    mcp_server_url: Optional[str] = None,
    mcp_tools_info: Optional[str] = None,
    mcp_api_key: Optional[str] = None,
    mcp_headers: Optional[str] = None,
    mcp_config: Optional[str] = None
) -> str:
    """Execute a single-shot goal or instruction autonomously."""
    proj_path = Path(project_dir).resolve()
    if not proj_path.exists() or not proj_path.is_dir():
        console.print(f"[bold red]Error: Project directory '{project_dir}' does not exist.[/bold red]")
        sys.exit(1)

    set_active_project_dir(proj_path)
    ensure_compile_commands(proj_path)
    set_require_permission(not auto_approve)

    system_prompt = load_system_prompt(system_prompt_file)

    if not model_name:
        model_name = "llama3.1:8b" if provider.lower() == "ollama" else "gemini-2.5-flash"

    prompt_source = f"File: {Path(system_prompt_file).name}" if system_prompt_file else "Default Engineering Principles"

    # Load user-configured MCP tools if provided
    mcp_tools: List[Any] = []
    if mcp_config or mcp_server_url:
        try:
            mcp_tools = build_mcp_tools(
                server_url=mcp_server_url,
                tools_info=mcp_tools_info,
                api_key=mcp_api_key,
                headers=mcp_headers,
                config_path=mcp_config
            )
            console.print(f"  [bold green][MCP][/bold green] Successfully connected. Loaded {len(mcp_tools)} MCP tool(s).")
        except Exception as e:
            console.print(f"  [bold red][MCP Error][/bold red] Failed to load MCP tools: {e}")

    effective_system_prompt = system_prompt
    if mcp_tools:
        tool_names = ", ".join(t.name for t in mcp_tools)
        effective_system_prompt = (
            f"{system_prompt}\n"
            f"════════════════════════════════════════════════════════════════════════════════\n"
            f"USER-CONFIGURED MCP TOOLS:\n"
            f"You are also equipped with the following custom MCP tools: {tool_names}.\n"
            f"Use them whenever relevant to fulfill user directives.\n"
            f"════════════════════════════════════════════════════════════════════════════════\n"
        )

    console.print(Panel(
        f"[bold cyan]C++ General Coding Assistant (Autonomous Task)[/bold cyan]\n"
        f"Project Path    : [yellow]{proj_path}[/yellow]\n"
        f"Directive       : [white]{prompt}[/white]\n"
        f"Provider        : [green]{provider}[/green]\n"
        f"Model           : [green]{model_name}[/green]\n"
        f"System Prompt   : [green]{prompt_source}[/green]\n"
        f"MCP Tools       : [blue]{len(mcp_tools)} active[/blue]\n"
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

    all_active_tools = list(CODING_ASSISTANT_TOOLS) + list(mcp_tools)

    app = build_coding_assistant_graph(
        llm=llm,
        tools=all_active_tools,
        max_context_tokens=max_context_tokens,
        summarize_threshold=summarize_threshold,
        system_prompt=effective_system_prompt
    )

    initial_messages: List[BaseMessage] = [
        SystemMessage(content=effective_system_prompt),
        HumanMessage(content=prompt)
    ]

    result = app.invoke(
        {"messages": initial_messages},
        {"recursion_limit": max_steps, "max_concurrency": 1}
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
        "--system-prompt-file",
        "-spf",
        type=str,
        default=None,
        help="Path to a custom system prompt / instructions file. Preserved permanently and never summarized."
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
    parser.add_argument(
        "--mcp-server-url",
        type=str,
        default=None,
        help="URL of an external or local MCP server (HTTP POST JSON-RPC 2.0 endpoint)."
    )
    parser.add_argument(
        "--mcp-tools",
        type=str,
        default=None,
        help="Path to MCP tools JSON file, directory of schemas, or 'auto' to auto-discover via tools/list."
    )
    parser.add_argument(
        "--mcp-api-key",
        type=str,
        default=None,
        help="API key or auth token for the MCP server (sent in X-API-Key and Authorization headers)."
    )
    parser.add_argument(
        "--mcp-headers",
        type=str,
        default=None,
        help="JSON string of custom HTTP headers to send to the MCP server."
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to an MCP configuration JSON file defining custom MCP servers."
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
            auto_approve=args.auto_approve,
            system_prompt_file=args.system_prompt_file,
            mcp_server_url=args.mcp_server_url,
            mcp_tools_info=args.mcp_tools,
            mcp_api_key=args.mcp_api_key,
            mcp_headers=args.mcp_headers,
            mcp_config=args.mcp_config
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
            auto_approve=args.auto_approve,
            system_prompt_file=args.system_prompt_file,
            mcp_server_url=args.mcp_server_url,
            mcp_tools_info=args.mcp_tools,
            mcp_api_key=args.mcp_api_key,
            mcp_headers=args.mcp_headers,
            mcp_config=args.mcp_config
        )
