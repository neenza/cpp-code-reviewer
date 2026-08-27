#!/usr/bin/env python3
"""
Interactive C++ Codebase Explainer & Tutor Agent.
Dedicated solely to investigating, explaining, and helping developers deeply understand a codebase.
Uses semantic clangd-query AST intelligence, ripgrep (rg), and bounded file reading.
Supports progressive adaptation to the user's level of technical understanding.
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

# LangChain / LangGraph imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Shared C++ Analysis Tools
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

load_dotenv()
console = Console()

# ============================================================================
# Codebase Explainer Persona & System Prompt
# ============================================================================

EXPLAINER_SYSTEM_PROMPT = """You are an expert C++ Software Architect and Codebase Mentor whose SOLE PURPOSE is to help the user deeply understand the given C++ codebase.

════════════════════════════════════════════════════════════════════════════════
CORE DIRECTIVES (WHAT YOU DO AND DO NOT DO):
1. ✅ INVESTIGATE & EXPLAIN: Your mission is to explore the existing code using your tools (clangd_query, ripgrep_search, read_project_file) and explain how the system works.
2. ❌ NO CODE WRITING OR DEBUGGING: You do NOT generate code implementations, fix bugs, write documentation files, or write unit tests. If asked to write new features or debug, clarify the existing logic instead.
3. 🎯 GROUNDED CITATIONS: Always ground your explanations in the actual codebase. Mention exact file paths, line numbers (`file.cpp:42-50`), class names, and method signatures found via tools.
4. 🧠 PROGRESSIVE & ADAPTIVE EXPLANATION:
   - Always assess the user's technical background and familiarity with the codebase.
   - If the user asks a high-level question, start with a clear architectural mental model, component diagram, or high-level narrative.
   - If the user asks deep technical questions, dive straight into concurrency models, memory lifecycles, RAII semantics, and cache-friendly designs.
   - Ask clarifying check-ins if appropriate: "Would you like me to trace how `OrderRepository` handles concurrent writes in detail?"
════════════════════════════════════════════════════════════════════════════════

TOOLS AT YOUR DISPOSAL:
- `clangd_query(command='search'|'show'|'usages'|'hierarchy'|'signature'|'interface', symbol_or_query='...')`:
  Deep semantic C++ code intelligence across classes, methods, inheritance, and call sites.
- `ripgrep_search(pattern='...', path_filter='...')`:
  Fast regex search across files for configurations, constants, synchronization, or keyword patterns.
- `read_project_file(file_path='...', start_line=..., end_line=...)`:
  Inspect exact file contents and implementation details.

EXPLANATION FORMATTING TIPS:
- Use clean Markdown with headers (`##`, `###`), bolding, and bullet points.
- Use ASCII or Mermaid diagrams when illustrating workflows, call sequences, or class hierarchies.
- Highlight key design patterns (e.g. Factory, Repository, Strategy, RAII, PIMPL).
"""

# ============================================================================
# Explainer Graph Builder
# ============================================================================

def build_explainer_graph(llm, tools=COMMON_CPP_TOOLS):
    """
    Build the LangGraph conversational assistant workflow.
    """
    llm_with_tools = llm.bind_tools(tools)

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

    tool_node = ToolNode(tools)

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent_step)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition, ["tools", END])
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# ============================================================================
# Interactive Session Runner
# ============================================================================

def run_interactive_explainer(
    project_dir: str,
    provider: str = "gemini",
    model_name: Optional[str] = None,
    ollama_host: str = "http://localhost:11434",
    max_steps: int = 60
) -> None:
    """
    Start the interactive codebase explainer REPL session.
    """
    proj_path = Path(project_dir).resolve()
    if not proj_path.exists() or not proj_path.is_dir():
        console.print(f"[bold red]Error: Project directory '{project_dir}' does not exist.[/bold red]")
        sys.exit(1)

    set_active_project_dir(proj_path)
    ensure_compile_commands(proj_path)

    if not model_name:
        if provider.lower() == "ollama":
            model_name = "llama3.1:8b"
        else:
            model_name = "gemini-3.5-flash-lite"

    console.print(Panel(
        f"[bold cyan]Interactive C++ Codebase Explainer & Tutor[/bold cyan]\n"
        f"Project Path : [yellow]{proj_path}[/yellow]\n"
        f"Provider     : [green]{provider}[/green]\n"
        f"Model        : [green]{model_name}[/green]\n"
        f"Max Steps/Turn: [yellow]{max_steps}[/yellow]\n"
        f"Tools Active : [blue]clangd-query, ripgrep (rg), read_project_file, list_project_structure[/blue]\n\n"
        f"[bold white]Available Shortcuts:[/bold white]\n"
        f"  [cyan]/overview[/cyan]           - Summarize codebase architecture, key components & entry points\n"
        f"  [cyan]/explore <symbol>[/cyan]   - Deep dive into a class, struct, or interface\n"
        f"  [cyan]/flow <function>[/cyan]    - Trace execution flow and call hierarchy for a function\n"
        f"  [cyan]/clear[/cyan]              - Reset conversation context\n"
        f"  [cyan]/help[/cyan]               - Show command tips\n"
        f"  [cyan]/exit[/cyan] or [cyan]quit[/cyan]       - Exit the session",
        title="Codebase Explainer Agent",
        border_style="cyan"
    ))

    llm = get_llm(provider=provider, model_name=model_name, ollama_host=ollama_host)
    app = build_explainer_graph(llm=llm, tools=COMMON_CPP_TOOLS)

    conversation_history: List[BaseMessage] = [
        SystemMessage(content=EXPLAINER_SYSTEM_PROMPT)
    ]

    console.print("[dim]Type your question or a slash command below (e.g. 'explain how orders are created' or '/overview').[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session ended. Goodbye![/yellow]")
            break

        if not user_input:
            continue

        # Handle slash commands and exit shortcuts
        if user_input.lower() in ("/exit", "exit", "quit", ":q"):
            console.print("[yellow]Exiting Codebase Explainer. Happy coding![/yellow]")
            break

        if user_input.lower() == "/clear":
            conversation_history = [SystemMessage(content=EXPLAINER_SYSTEM_PROMPT)]
            console.print("[bold cyan]✔ Conversation history cleared.[/bold cyan]\n")
            continue

        if user_input.lower() == "/help":
            console.print(Panel(
                "[bold cyan]Interactive Explainer Help & Commands[/bold cyan]\n\n"
                "• [bold]/overview[/bold]: Inspects project layout and CMake to provide an architectural tour.\n"
                "• [bold]/explore <symbol>[/bold]: Deeply analyzes a class, its members, inheritance, and usages.\n"
                "• [bold]/flow <function>[/bold]: Traces execution call trees and data flow.\n"
                "• [bold]/clear[/bold]: Resets context memory for a brand new topic.\n"
                "• [bold]Natural questions[/bold]: Ask anything (e.g., 'how does memory management work in SessionManager?', 'is this thread-safe?').",
                border_style="blue"
            ))
            continue

        # Transform slash command shorthands into targeted prompts
        if user_input.startswith("/overview"):
            effective_prompt = (
                "Please give a comprehensive architectural overview of this codebase.\n"
                "1. Start by calling 'list_project_structure' and reading 'CMakeLists.txt' (or top-level build config).\n"
                "2. Inspect the key header files in the core modules.\n"
                "3. Explain the main components, their responsibilities, and how data/control flows between them."
            )
        elif user_input.startswith("/explore"):
            symbol = user_input[len("/explore"):].strip()
            if not symbol:
                console.print("[yellow]Usage: /explore <SymbolName> (e.g., /explore OrderRepository)[/yellow]")
                continue
            effective_prompt = (
                f"Please conduct an in-depth investigation of the symbol '{symbol}' using clangd_query "
                f"(show, interface, hierarchy, usages) and explain: "
                f"1. What it is and what role it plays in the architecture. "
                f"2. Its public interface, fields, and dependencies. "
                f"3. Where and how it is used across the codebase."
            )
        elif user_input.startswith("/flow"):
            func = user_input[len("/flow"):].strip()
            if not func:
                console.print("[yellow]Usage: /flow <FunctionName> (e.g., /flow process_order_payment)[/yellow]")
                continue
            effective_prompt = (
                f"Please trace the execution flow and call chain for '{func}'. "
                f"Use clangd_query and ripgrep to find caller sites and downstream dependencies, "
                f"and explain step-by-step how data moves through this workflow."
            )
        else:
            effective_prompt = user_input

        # Append user message
        conversation_history.append(HumanMessage(content=effective_prompt))

        console.print(f"\n[dim cyan]🔍 Investigating codebase...[/dim cyan]")

        # Stream graph execution
        current_state = {"messages": conversation_history}
        assistant_reply = ""

        try:
            for step in app.stream(current_state, {"recursion_limit": max_steps}, stream_mode="updates"):
                for node_name, node_update in step.items():
                    if node_name == "agent":
                        msg = node_update["messages"][-1]
                        conversation_history.append(msg)
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                console.print(f"  [magenta]▶ Tool:[/magenta] [cyan]{escape(tc['name'])}[/cyan]({escape(json.dumps(tc['args']))})")
                        else:
                            assistant_reply = extract_text(msg.content)
                    elif node_name == "tools":
                        for msg in node_update["messages"]:
                            conversation_history.append(msg)
                            raw_text = extract_text(msg.content)
                            preview = raw_text[:120].replace("\n", " ")
                            if len(raw_text) > 120:
                                preview += "..."
                            console.print(f"[dim]    ↳ Result: {escape(preview)}[/dim]")
        except Exception as e:
            # If recursion limit is reached or interrupted, request model to synthesize from gathered context
            console.print(f"[dim yellow]  (Exploration reached turn limit of {max_steps} steps; compiling summary...)[/dim yellow]")
            try:
                summary_prompt = HumanMessage(content="Please synthesize your complete architectural explanation now based on all the files and symbols you explored above.")
                recovery_messages = conversation_history + [summary_prompt]
                final_response = llm.invoke(recovery_messages)
                assistant_reply = extract_text(final_response.content)
                conversation_history.append(final_response)
            except Exception as final_e:
                assistant_reply = f"Exploration completed. (Encountered: {e})"

        # Render assistant response with rich markdown
        console.print("\n" + "─"*80)
        console.print(Panel(
            Markdown(assistant_reply or "I investigated the code but have no additional notes."),
            title="[bold cyan]Codebase Explainer[/bold cyan]",
            border_style="cyan"
        ))
        console.print("─"*80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive C++ Codebase Explainer & Tutor Agent using LangGraph, clangd-query, and ripgrep."
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
        "--max-steps",
        type=int,
        default=60,
        help="Maximum tool execution steps per interaction turn (default: 60)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_interactive_explainer(
        project_dir=args.project_dir,
        provider=args.provider,
        model_name=args.model,
        ollama_host=args.ollama_host,
        max_steps=args.max_steps
    )
