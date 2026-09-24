"""
Shared C++ Analysis Tools and Utilities for Autonomous & Interactive Agents.
Integrates clangd-query, ripgrep (rg), bounded file readers, and LLM factories.
"""

import os
import sys
import re
import json
import time
import shutil
import subprocess
from typing import Literal, Optional, List, Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage

load_dotenv()
console = Console()

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

    while split_idx > 0:
        if isinstance(messages[split_idx], ToolMessage):
            split_idx -= 1
            continue
        if isinstance(messages[split_idx - 1], AIMessage) and getattr(messages[split_idx - 1], "tool_calls", None):
            split_idx -= 1
            continue
        break

    older = messages[:split_idx]
    recent = messages[split_idx:]
    return older, recent


_ACTIVE_PROJECT_DIR: Path = Path.cwd()
_REQUIRE_PERMISSION: bool = True
_PERMISSION_CALLBACK: Optional[Any] = None


def set_require_permission(val: bool) -> None:
    """Enable or disable interactive permission checks for file writing, editing, and shell execution."""
    global _REQUIRE_PERMISSION
    _REQUIRE_PERMISSION = val


def get_require_permission() -> bool:
    """Check if permission is currently required for destructive actions."""
    global _REQUIRE_PERMISSION
    return _REQUIRE_PERMISSION


def set_permission_callback(cb: Optional[Any]) -> None:
    """Set a custom callback function (prompt_text: str) -> bool for permission checks."""
    global _PERMISSION_CALLBACK
    _PERMISSION_CALLBACK = cb


class PermissionResult:
    """Result of an interactive permission query, containing approval status and optional rejection reason."""
    def __init__(self, allowed: bool, reason: str = ""):
        self.allowed = bool(allowed)
        self.reason = str(reason).strip() if reason else ""

    def __bool__(self) -> bool:
        return self.allowed

    def __iter__(self):
        return iter((self.allowed, self.reason))

    def __getitem__(self, index: int):
        return (self.allowed, self.reason)[index]

    def __len__(self) -> int:
        return 2

    def __repr__(self) -> str:
        return f"PermissionResult(allowed={self.allowed}, reason={self.reason!r})"


def _parse_permission_input(raw: str) -> tuple[bool, str]:
    """Parse user terminal response into (allowed: bool, reason: str)."""
    ans = raw.strip()
    if ans.lower() in ("y", "yes", "allow", "approve"):
        return True, ""
    if ans.lower() in ("n", "no", "deny", "reject", ""):
        return False, ""

    # Strip common leading denial prefixes if present
    reason = ans
    for pfx in ("no, ", "no: ", "no - ", "n, ", "n: ", "n - ", "reject: ", "reject, ", "deny: ", "deny - "):
        if reason.lower().startswith(pfx):
            reason = reason[len(pfx):].strip()
            break
    return False, reason


def ask_user_permission(prompt_text: str) -> PermissionResult:
    """Prompt the user for permission to execute a file modification or shell execution.
    
    Accepts:
      - 'y' / 'yes' to approve.
      - 'n' / 'no' or Enter to reject without feedback.
      - Any other text as an explicit rejection reason/feedback passed back to the model.
    """
    global _REQUIRE_PERMISSION, _PERMISSION_CALLBACK
    if not _REQUIRE_PERMISSION:
        return PermissionResult(True, "")

    if _PERMISSION_CALLBACK is not None:
        try:
            res = _PERMISSION_CALLBACK(prompt_text)
            if isinstance(res, PermissionResult):
                return res
            if isinstance(res, tuple) and len(res) == 2:
                return PermissionResult(bool(res[0]), str(res[1]))
            if isinstance(res, bool):
                return PermissionResult(res, "")
            if isinstance(res, str):
                allowed, reason = _parse_permission_input(res)
                return PermissionResult(allowed, reason)
            return PermissionResult(bool(res), "")
        except Exception:
            return PermissionResult(False, "")

    try:
        from rich.prompt import Prompt
        prompt_formatted = (
            f"[bold cyan]{prompt_text}[/bold cyan] "
            f"[dim][[bold green]y[/bold green] to allow, [bold red]n[/bold red] or [yellow]<reason>[/yellow] to reject][/dim]"
        )
        raw_ans = Prompt.ask(prompt_formatted, default="n")
    except (KeyboardInterrupt, EOFError):
        return PermissionResult(False, "")
    except Exception:
        try:
            raw_ans = input(f"{prompt_text} [y to allow, n or <reason> to reject]: ").strip()
        except (KeyboardInterrupt, EOFError):
            return PermissionResult(False, "")

    allowed, reason = _parse_permission_input(raw_ans)
    return PermissionResult(allowed, reason)


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


def parse_interface_methods(interface_text: str, default_class: str = "") -> List[Dict[str, Any]]:
    """
    Parse method signatures and names from clangd-query interface output.
    Distinguishes trivial methods (= default, = delete, = 0) from non-trivial methods
    that require body inspection via 'clangd_query show'.
    """
    results = []
    lines = interface_text.splitlines()
    in_interface = False
    class_name = default_class

    class_header_pattern = re.compile(r'(?:class|struct)\s+(?:[A-Za-z0-9_]+::)*([A-Za-z0-9_]+)')
    func_pattern = re.compile(r'(?:(~[A-Za-z0-9_]+)|([A-Za-z0-9_]+)|(operator\s*[^\(\s]+))\s*\(')
    control_keywords = {'if', 'for', 'while', 'switch', 'catch'}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not class_name:
            cm = class_header_pattern.search(stripped)
            if cm:
                class_name = cm.group(1)

        if "Public Interface:" in stripped:
            in_interface = True
            continue
        if not in_interface:
            continue

        if (stripped.startswith("//") or stripped.startswith("/*") or
            stripped.startswith("*") or stripped.startswith("Critical") or
            stripped.startswith("Minor") or stripped.startswith("Non-copyable") or
            stripped.startswith("Note:") or stripped.startswith("#")):
            continue

        matches = list(func_pattern.finditer(stripped))
        if matches:
            last_match = matches[-1]
            func_name = (last_match.group(1) or last_match.group(2) or last_match.group(3) or "").strip()
            if not func_name or func_name in control_keywords:
                continue

            is_trivial = bool(re.search(r'=\s*(?:default|delete|0)\s*;?$', stripped))
            full_sym = f"{class_name}::{func_name}" if class_name else func_name
            results.append({
                "class_name": class_name,
                "method_name": func_name,
                "full_symbol": full_sym,
                "is_trivial": is_trivial,
                "signature": stripped
            })

    return results


def discover_module_functions(project_dir: Path, files: List[str]) -> List[str]:
    """Statically discover function definitions across given files in a module."""
    control_keywords = {'if', 'for', 'while', 'switch', 'catch', 'sizeof', 'decltype', 'return'}
    pattern = re.compile(
        r'^\s*(?:[A-Za-z0-9_<>:,\s\*&]+?\s+)?([A-Za-z0-9_]+::[~A-Za-z0-9_]+|[A-Za-z0-9_]+)\s*\([^;{}]*\)\s*(?:const)?\s*(?:noexcept)?\s*\{',
        re.MULTILINE
    )
    funcs = set()
    for rel_path in files:
        full_path = project_dir / rel_path
        if not full_path.exists() or full_path.is_dir():
            continue
        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
            for m in pattern.finditer(content):
                name = m.group(1).strip()
                if name not in control_keywords and not name.startswith("std::"):
                    funcs.add(name)
        except Exception:
            pass
    return sorted(list(funcs))


def extract_touched_files(text: str, project_files: List[str]) -> List[str]:
    """Extract which project files were referenced or displayed in tool output."""
    touched = set()
    for pf in project_files:
        if pf in text:
            touched.add(pf)
        else:
            base = Path(pf).name
            if len(base) > 4 and re.search(r'\b' + re.escape(base) + r'\b', text):
                touched.add(pf)
    return sorted(list(touched))


COMMON_CPP_TOOLS = [clangd_query, ripgrep_search, read_project_file, list_project_structure]


@tool
def write_project_file(
    file_path: str,
    content: str,
    overwrite: bool = True
) -> str:
    """Write or overwrite a file in the project directory.
    Prompts the user for interactive permission before modifying the filesystem.

    Args:
        file_path: Relative path to the file to create or overwrite.
        content: The text content to write into the file.
        overwrite: Whether to overwrite the file if it already exists (default True).
    """
    global _ACTIVE_PROJECT_DIR
    if not file_path:
        return "Error: 'file_path' is required."

    target = (_ACTIVE_PROJECT_DIR / file_path).resolve()
    if not str(target).startswith(str(_ACTIVE_PROJECT_DIR)):
        return f"Error: Access denied. Cannot write outside project directory: {file_path}"

    if target.exists() and not overwrite:
        return f"Error: File '{file_path}' already exists and overwrite is set to False."

    action_str = "overwrite" if target.exists() else "create"
    lines = content.count("\n") + (1 if content else 0)

    console.print(f"\n  [bold yellow][Permission Request][/bold yellow] Agent requests permission to {action_str} file: [cyan]{escape(file_path)}[/cyan] ({lines} lines, {len(content):,} chars)")
    prompt_msg = f"Do you allow the agent to {action_str} '{file_path}'?"
    perm = ask_user_permission(prompt_msg)
    if not perm.allowed:
        if perm.reason:
            console.print(f"  [bold red][Permission Denied][/bold red] User rejected {action_str} for '{escape(file_path)}' with reason: [yellow]{escape(perm.reason)}[/yellow]")
            return (
                f"Permission Denied: User rejected request to {action_str} file '{file_path}'.\n"
                f"User Rejection Reason / Feedback: {perm.reason}\n"
                f"Please carefully inspect the feedback above and adjust your implementation accordingly."
            )
        console.print(f"  [bold red][Permission Denied][/bold red] User rejected {action_str} for '{escape(file_path)}'")
        return f"Permission Denied: User did not grant permission to {action_str} '{file_path}'."

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        console.print(f"  [bold green][OK][/bold green] Successfully wrote '{escape(file_path)}' ({lines} lines)")
        return f"Successfully {action_str}d file '{file_path}' ({lines} lines, {len(content)} characters)."
    except Exception as e:
        return f"Error writing file '{file_path}': {e}"


@tool
def edit_project_file(
    file_path: str,
    target_content: str,
    replacement_content: str,
    allow_multiple: bool = False
) -> str:
    """Perform a precise, surgical search-and-replace edit on an existing file.
    Prompts the user for interactive permission before applying the patch.

    Args:
        file_path: Relative path to the file to modify.
        target_content: The exact string block within the file to replace (must match whitespace and indentation exactly).
        replacement_content: The new replacement string block.
        allow_multiple: If True, replaces all occurrences of target_content; if False, errors if target_content appears more than once (default False).
    """
    global _ACTIVE_PROJECT_DIR
    if not file_path:
        return "Error: 'file_path' is required."
    if not target_content:
        return "Error: 'target_content' cannot be empty."

    target = (_ACTIVE_PROJECT_DIR / file_path).resolve()
    if not str(target).startswith(str(_ACTIVE_PROJECT_DIR)):
        return f"Error: Access denied. Cannot edit outside project directory: {file_path}"

    if not target.exists():
        return f"Error: File '{file_path}' does not exist."
    if target.is_dir():
        return f"Error: '{file_path}' is a directory, not a file."

    try:
        content = target.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file '{file_path}': {e}"

    occurrences = content.count(target_content)
    if occurrences == 0:
        return (
            f"Error: target_content not found in '{file_path}'. "
            f"Please verify exact indentation and line breaks, or inspect the file with read_project_file first."
        )

    if occurrences > 1 and not allow_multiple:
        return (
            f"Error: target_content matched {occurrences} times in '{file_path}'. "
            f"To prevent unintended edits, provide more surrounding context lines to make the match unique, "
            f"or set allow_multiple=True."
        )

    target_lines = target_content.splitlines()
    repl_lines = replacement_content.splitlines()
    console.print(f"\n  [bold yellow][Permission Request][/bold yellow] Agent requests permission to patch file: [cyan]{escape(file_path)}[/cyan] ({occurrences} match(es))")
    console.print("  [red]--- Target Content ---[/red]")
    for line in target_lines[:10]:
        console.print(f"  [red]- {escape(line)}[/red]")
    if len(target_lines) > 10:
        console.print(f"  [dim red]  ... ({len(target_lines) - 10} more lines)[/dim red]")
    console.print("  [green]+++ Replacement Content +++[/green]")
    for line in repl_lines[:10]:
        console.print(f"  [green]+ {escape(line)}[/green]")
    if len(repl_lines) > 10:
        console.print(f"  [dim green]  ... ({len(repl_lines) - 10} more lines)[/dim green]")

    prompt_msg = f"Do you allow the agent to apply this patch to '{file_path}'?"
    perm = ask_user_permission(prompt_msg)
    if not perm.allowed:
        if perm.reason:
            console.print(f"  [bold red][Permission Denied][/bold red] User rejected patch for '{escape(file_path)}' with reason: [yellow]{escape(perm.reason)}[/yellow]")
            return (
                f"Permission Denied: User rejected the proposed patch for '{file_path}'.\n"
                f"User Rejection Reason / Feedback: {perm.reason}\n"
                f"Please carefully inspect the feedback above and adjust your patch or approach accordingly."
            )
        console.print(f"  [bold red][Permission Denied][/bold red] User rejected patch for '{escape(file_path)}'")
        return f"Permission Denied: User did not grant permission to apply patch to '{file_path}'."

    new_content = content.replace(target_content, replacement_content, -1 if allow_multiple else 1)
    try:
        target.write_text(new_content, encoding="utf-8")
        console.print(f"  [bold green][OK][/bold green] Successfully patched '{escape(file_path)}'")
        return f"Successfully applied patch to '{file_path}' ({occurrences} occurrence(s) replaced)."
    except Exception as e:
        return f"Error writing patched file '{file_path}': {e}"


@tool
def execute_shell_command(
    command: str,
    timeout: int = 60
) -> str:
    """Execute a shell command (such as cmake, make, ctest, clang-format, git status/diff)
    inside the active project directory. Prompts the user for interactive permission before executing.

    Args:
        command: The shell command string to execute.
        timeout: Maximum execution time in seconds (default 60).
    """
    global _ACTIVE_PROJECT_DIR
    if not command or not command.strip():
        return "Error: 'command' cannot be empty."

    cmd_stripped = command.strip()
    console.print(f"\n  [bold yellow][Permission Request][/bold yellow] Agent requests permission to execute command:")
    console.print(f"    [bold cyan]{escape(cmd_stripped)}[/bold cyan] (cwd: {_ACTIVE_PROJECT_DIR.name})")

    prompt_msg = f"Do you allow the agent to execute shell command: '{cmd_stripped}'?"
    perm = ask_user_permission(prompt_msg)
    if not perm.allowed:
        if perm.reason:
            console.print(f"  [bold red][Permission Denied][/bold red] User rejected command: '{escape(cmd_stripped)}' with reason: [yellow]{escape(perm.reason)}[/yellow]")
            return (
                f"Permission Denied: User rejected the execution of shell command '{cmd_stripped}'.\n"
                f"User Rejection Reason / Feedback: {perm.reason}\n"
                f"Please carefully inspect the feedback above and adjust or cancel your command accordingly."
            )
        console.print(f"  [bold red][Permission Denied][/bold red] User rejected command: '{escape(cmd_stripped)}'")
        return f"Permission Denied: User did not grant permission to execute command: '{cmd_stripped}'."

    try:
        console.print(f"  [dim]Running: {escape(cmd_stripped)}...[/dim]")
        res = subprocess.run(
            cmd_stripped,
            shell=True,
            cwd=str(_ACTIVE_PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        stdout = res.stdout.strip()
        stderr = res.stderr.strip()
        code = res.returncode

        status_tag = "[bold green][OK][/bold green]" if code == 0 else f"[bold red][Exit {code}][/bold red]"
        console.print(f"  {status_tag} Command completed with exit code {code}")

        if stdout:
            lines = stdout.splitlines()
            if len(lines) > 80:
                display_stdout = "\n".join(lines[:40]) + f"\n\n[dim]... [Truncated {len(lines) - 80} lines in terminal preview] ...[/dim]\n\n" + "\n".join(lines[-40:])
            else:
                display_stdout = stdout
            console.print(Panel(
                escape(display_stdout),
                title=f"STDOUT: [bold cyan]{escape(cmd_stripped)}[/bold cyan]",
                border_style="dim green" if code == 0 else "yellow"
            ))

        if stderr:
            lines = stderr.splitlines()
            if len(lines) > 80:
                display_stderr = "\n".join(lines[:40]) + f"\n\n[dim]... [Truncated {len(lines) - 80} lines in terminal preview] ...[/dim]\n\n" + "\n".join(lines[-40:])
            else:
                display_stderr = stderr
            console.print(Panel(
                escape(display_stderr),
                title=f"STDERR: [bold red]{escape(cmd_stripped)}[/bold red]",
                border_style="bold red"
            ))

        if not stdout and not stderr:
            console.print("  [dim](Command produced no output)[/dim]")

        output_parts = [f"Exit code: {code}"]
        if stdout:
            lines = stdout.splitlines()
            if len(lines) > 80:
                stdout_for_llm = "\n".join(lines[:80]) + f"\n... [Truncated {len(lines) - 80} lines]"
            else:
                stdout_for_llm = stdout
            output_parts.append(f"STDOUT:\n{stdout_for_llm}")
        if stderr:
            lines = stderr.splitlines()
            if len(lines) > 80:
                stderr_for_llm = "\n".join(lines[:80]) + f"\n... [Truncated {len(lines) - 80} lines]"
            else:
                stderr_for_llm = stderr
            output_parts.append(f"STDERR:\n{stderr_for_llm}")
        if not stdout and not stderr:
            output_parts.append("(Command produced no output)")

        return "\n\n".join(output_parts)
    except subprocess.TimeoutExpired:
        console.print(f"  [bold red][Timeout][/bold red] Command '{escape(cmd_stripped)}' timed out after {timeout}s")
        return f"Error: Command '{cmd_stripped}' timed out after {timeout} seconds."
    except Exception as e:
        console.print(f"  [bold red][Error][/bold red] Error executing command '{escape(cmd_stripped)}': {e}")
        return f"Error executing command '{cmd_stripped}': {e}"


CODING_ASSISTANT_TOOLS = [
    clangd_query,
    ripgrep_search,
    read_project_file,
    list_project_structure,
    write_project_file,
    edit_project_file,
    execute_shell_command
]

__all__ = [
    "clangd_query",
    "ripgrep_search",
    "read_project_file",
    "list_project_structure",
    "COMMON_CPP_TOOLS",
    "write_project_file",
    "edit_project_file",
    "execute_shell_command",
    "CODING_ASSISTANT_TOOLS",
    "set_active_project_dir",
    "get_active_project_dir",
    "ensure_compile_commands",
    "get_llm",
    "extract_text",
    "extract_message_text",
    "discover_project_classes",
    "group_classes_by_module",
    "parse_interface_methods",
    "discover_module_functions",
    "extract_touched_files",
    "count_tokens",
    "count_message_tokens",
    "partition_messages_safely",
    "set_require_permission",
    "get_require_permission",
    "set_permission_callback",
    "ask_user_permission",
    "PermissionResult",
]
