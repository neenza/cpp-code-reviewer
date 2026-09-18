"""
Shared C++ Analysis Tools and Utilities for Autonomous & Interactive Agents.
Integrates clangd-query, ripgrep (rg), bounded file readers, and LLM factories.
"""

import os
import sys
import json
import time
import shutil
import subprocess
from typing import Literal, Optional, List, Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, BaseMessage

load_dotenv()
console = Console()

_ACTIVE_PROJECT_DIR: Path = Path.cwd()


def set_active_project_dir(project_dir: Path) -> None:
    """Set the active project directory for tool executions."""
    global _ACTIVE_PROJECT_DIR
    _ACTIVE_PROJECT_DIR = project_dir.resolve()


def get_active_project_dir() -> Path:
    """Get the active project directory."""
    global _ACTIVE_PROJECT_DIR
    return _ACTIVE_PROJECT_DIR


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


@tool
def clangd_query(
    command: str = "show",
    symbol_or_query: Optional[str] = None,
    query: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: Optional[int] = None,
    **kwargs: Any
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

    resolved = symbol_or_query or query or symbol or kwargs.get("name") or kwargs.get("target") or ""
    if not resolved:
        return "Error: 'symbol_or_query' argument is required for clangd-query (e.g. clangd_query(command='show', symbol_or_query='MyClass'))."

    cmd_str = command if command in ["search", "show", "usages", "hierarchy", "signature", "interface"] else "show"
    cmd = ["clangd-query", cmd_str, resolved]
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
            return f"[clangd-query {cmd_str} '{resolved}']: No output returned."
        return output
    except FileNotFoundError:
        return "Error: 'clangd-query' executable not found in PATH."
    except subprocess.TimeoutExpired:
        return f"Error: 'clangd-query {cmd_str} {resolved}' timed out after 30 seconds."
    except Exception as e:
        return f"Error executing clangd-query: {e}"


@tool
def ripgrep_search(
    pattern: Optional[str] = None,
    query: Optional[str] = None,
    path_filter: Optional[str] = None,
    case_insensitive: bool = False,
    is_regex: bool = True,
    file_names_only: bool = False,
    max_results: int = 40,
    **kwargs: Any
) -> str:
    """Search codebase text using ripgrep (rg).
    Ideal for:
      - Locating patterns like raw pointers, malloc/free, new/delete, strcpy/sprintf, reinterpret_cast
      - Checking for synchronization primitives (mutex, lock_guard, shared_mutex, atomic)
      - Finding preprocessor directives, CMake definitions, include statements, or comments
    """
    global _ACTIVE_PROJECT_DIR

    resolved = pattern or query or kwargs.get("search") or kwargs.get("text") or ""
    if not resolved:
        return "Error: 'pattern' argument is required for ripgrep_search (e.g. ripgrep_search(pattern='mutex'))."

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
    cmd.extend(["--glob", "!build/**", "--glob", "!.cache/**", "--glob", "!third_party/**", "--glob", "!vendor/**"])

    cmd.append(resolved)

    resolved_filter = path_filter or kwargs.get("path") or kwargs.get("dir")
    if resolved_filter:
        cmd.append(resolved_filter)
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
            return f"[ripgrep '{resolved}']: No matches found in {resolved_filter or 'project'}."
        
        lines = output.splitlines()
        if len(lines) > max_results:
            truncated = "\n".join(lines[:max_results])
            return f"{truncated}\n... [Truncated {len(lines) - max_results} additional matches]"
        return output
    except FileNotFoundError:
        return "Error: 'rg' (ripgrep) executable not found in PATH."
    except subprocess.TimeoutExpired:
        return f"Error: ripgrep search for '{resolved}' timed out."
    except Exception as e:
        return f"Error executing ripgrep: {e}"


@tool
def read_project_file(
    file_path: Optional[str] = None,
    path: Optional[str] = None,
    filename: Optional[str] = None,
    filepath: Optional[str] = None,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    **kwargs: Any
) -> str:
    """Read contents of a file in the project (such as CMakeLists.txt, small configs, or targeted line ranges).
    NOTE: NEVER read entire large files (> 80 lines) directly. Instead, use 'clangd_query' with:
      - command='interface', symbol_or_query='<ClassName>' to inspect class layouts and public APIs.
      - command='show', symbol_or_query='<ClassName::MethodName>' to inspect method definitions.
      - command='usages', symbol_or_query='<SymbolName>' to inspect references across the project.
      - Or provide 'start_line' and 'end_line' to read specific focused ranges (up to 80 lines).
    Full file reads without line limits are strictly permitted only for small files (< 80 lines) or config files.

    Args:
      file_path: Relative path to the file to inspect (e.g. 'src/engine/worker.cpp'). Also accepts 'path' or 'filename'.
      start_line: Optional starting line number (1-indexed).
      end_line: Optional ending line number (1-indexed).
    """
    global _ACTIVE_PROJECT_DIR

    resolved = file_path or path or filename or filepath or kwargs.get("file") or kwargs.get("file_name") or ""
    if not resolved:
        return "Error: 'file_path' argument is required. Please specify the relative file path to read (e.g. read_project_file(file_path='src/main.cpp'))."

    target = (_ACTIVE_PROJECT_DIR / resolved).resolve()
    if not str(target).startswith(str(_ACTIVE_PROJECT_DIR)):
        return f"Error: Access denied. Cannot read outside project directory: {resolved}"

    if not target.exists():
        return f"Error: File '{resolved}' does not exist."
    if target.is_dir():
        return f"Error: '{resolved}' is a directory, not a file."

    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        is_config = target.name.lower() in ["cmakelists.txt", "conanfile.txt", "vcpkg.json"] or target.name.endswith(".json")

        # Prohibit reading full large files (> 80 lines)
        if start_line is None and end_line is None and total_lines > 80 and not is_config:
            selected = lines[:60]
            formatted = "".join(f"{i:4d} | {line}" for i, line in enumerate(selected, start=1))
            return (
                f"File: {resolved} (Showing lines 1-60 of {total_lines} total lines)\n\n{formatted}\n\n"
                f"[Notice: File '{resolved}' has {total_lines} lines. Reading large files entirely is prohibited to prevent context bloat. "
                f"Showing first 60 lines. Please prioritize using 'clangd_query(command='interface', symbol_or_query='<ClassName>')' "
                f"to inspect class structure, 'clangd_query(command='show', symbol_or_query='<MethodName>')' to inspect method implementations, "
                f"'clangd_query(command='usages', symbol_or_query='<SymbolName>')' for cross-file references, "
                f"or specify 'start_line' and 'end_line' (max 80 lines) for a targeted range.]"
            )

        start = max(1, start_line) if start_line is not None else 1
        end = min(total_lines, end_line) if end_line is not None else total_lines

        if start > total_lines:
            return f"Error: start_line {start} exceeds total lines ({total_lines})."

        # Cap line window if too wide
        if end - start + 1 > 100:
            end = start + 99
            capped_note = f"\n[Range capped at 100 lines to preserve context budget. Use clangd_query for semantic symbols.]"
        else:
            capped_note = ""

        selected = lines[start - 1:end]
        formatted = "".join(f"{i:4d} | {line}" for i, line in enumerate(selected, start=start))
        return f"File: {resolved} (Lines {start}-{end} of {total_lines})\n\n{formatted}{capped_note}"
    except Exception as e:
        return f"Error reading file '{resolved}': {e}"


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


def discover_project_classes(project_dir: Path, ignored_dirs: Optional[set] = None) -> Dict[str, List[str]]:
    """
    Scan project files using regex to discover all class and struct declarations across the project.
    Returns a dictionary mapping relative file path to list of declared class/struct names.
    """
    import re
    ignored = {"build", ".cache", ".git", ".vscode", ".idea", "third_party", "thirdparty", "external", "vendor"}
    if ignored_dirs:
        ignored.update(d.lower() for d in ignored_dirs)

    pattern = re.compile(r"^\s*(?:class|struct)\s+([A-Za-z0-9_]+)\b(?!\s*;)", re.MULTILINE)
    file_to_classes: Dict[str, List[str]] = {}

    for ext in ["*.h", "*.hpp", "*.hxx", "*.cpp", "*.cc", "*.cxx"]:
        for file_path in sorted(project_dir.glob(f"**/{ext}")):
            try:
                rel = file_path.relative_to(project_dir)
            except ValueError:
                continue
            if any(part.lower() in ignored or part.startswith(".") for part in rel.parts):
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                matches = pattern.findall(content)
                cleaned = [m for m in matches if m not in {"class", "struct", "void", "int", "bool", "char", "float", "double"}]
                if cleaned:
                    file_to_classes[str(rel)] = sorted(list(set(cleaned)))
            except Exception:
                pass

    return file_to_classes


def group_classes_by_module(file_to_classes: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Group discovered classes by parent module directory."""
    module_classes: Dict[str, List[str]] = {}
    for rel_file, classes in file_to_classes.items():
        mod = str(Path(rel_file).parent)
        if mod not in module_classes:
            module_classes[mod] = []
        module_classes[mod].extend(classes)
    for mod in module_classes:
        module_classes[mod] = sorted(list(set(module_classes[mod])))
    return module_classes


def get_llm(
    provider: str,
    model_name: str,
    ollama_host: str = "http://localhost:11434",
    max_context_tokens: int = 32000
):
    """
    Instantiate the appropriate LLM based on provider (Ollama or Gemini).
    For Ollama, passes num_ctx=max_context_tokens (default: 32000) so Ollama allocates
    a full 32k token context window instead of its default 2048 tokens.
    """
    provider = provider.lower()
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            console.print(f"[bold green]Initializing Ollama LLM:[/bold green] model='{model_name}', host='{ollama_host}', num_ctx={max_context_tokens:,}")
            return ChatOllama(
                model=model_name,
                base_url=ollama_host,
                temperature=0.1,
                num_ctx=max_context_tokens,
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


@tool
def list_project_structure(max_files_per_dir: int = 15) -> str:
    """List directory structure, modules, headers (.h, .hpp), and sources (.cpp, .cc) in the project.
    Ideal as the very first step to understand repository layout, directory hierarchy, and key modules.
    """
    global _ACTIVE_PROJECT_DIR
    proj_path = _ACTIVE_PROJECT_DIR

    ignored = {
        "build", ".cache", ".git", ".vscode", ".idea",
        "third_party", "thirdparty", "external", "vendor",
        "deps", "_deps", "vcpkg_installed", "conan", "submodules"
    }

    def is_ignored(p: Path) -> bool:
        return any(part.lower() in ignored or part.startswith(".") for part in p.parts)

    headers = [p.relative_to(proj_path) for p in sorted(proj_path.glob("**/*.h*")) if not is_ignored(p)]
    sources = [p.relative_to(proj_path) for p in sorted(proj_path.glob("**/*.c*")) if not is_ignored(p)]

    all_files = sorted(list(set(headers + sources)))
    dir_map: Dict[str, List[str]] = {}
    for f in all_files:
        parent = str(f.parent)
        if parent not in dir_map:
            dir_map[parent] = []
        dir_map[parent].append(str(f))

    lines = [f"Project Root: {proj_path.name}/ ({len(all_files)} C++ files across {len(dir_map)} directories)\n"]
    for d, files in sorted(dir_map.items()):
        lines.append(f"- {d}/ ({len(files)} files)")
        for f in files[:max_files_per_dir]:
            lines.append(f"   ├── {Path(f).name}")
        if len(files) > max_files_per_dir:
            lines.append(f"   └── ... ({len(files) - max_files_per_dir} more files)")
        lines.append("")

    return "\n".join(lines)


COMMON_CPP_TOOLS = [clangd_query, ripgrep_search, read_project_file, list_project_structure]

__all__ = [
    "clangd_query",
    "ripgrep_search",
    "read_project_file",
    "list_project_structure",
    "COMMON_CPP_TOOLS",
    "set_active_project_dir",
    "get_active_project_dir",
    "ensure_compile_commands",
    "get_llm",
    "extract_text",
    "extract_message_text",
    "discover_project_classes",
    "group_classes_by_module",
]
