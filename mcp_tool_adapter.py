#!/usr/bin/env python3
"""
Generic Model Context Protocol (MCP) Tool Adapter.
Provides a universal, non-hardcoded interface connecting any MCP server
(via HTTP JSON-RPC 2.0 or standard tools schemas) to LangChain and LangGraph.
"""

import os
import sys
import json
import uuid
import socket
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from pydantic import create_model, Field, BaseModel
from langchain_core.tools import StructuredTool

from cpp_agent_tools import (
    ask_user_permission,
    get_require_permission,
    PermissionResult
)

# Type mapping from JSON Schema types to Python types
JSON_TYPE_MAP: Dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


class MCPClient:
    """Generic JSON-RPC 2.0 client for communicating with any HTTP/HTTPS MCP server."""

    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 60,
        api_key: Optional[str] = None
    ) -> None:
        self.base_url = base_url.strip()
        self.timeout = timeout
        self.headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "Generic-MCP-Client/1.0"
        }
        if api_key:
            self.headers["X-API-Key"] = api_key
            self.headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            self.headers.update(headers)

    def send_rpc(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        req_id: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Dispatch a JSON-RPC 2.0 request to the MCP server endpoint."""
        if req_id is None:
            req_id = str(uuid.uuid4())

        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params if params is not None else {}
        }
        data_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            self.base_url,
            data=data_bytes,
            headers=self.headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_bytes = resp.read()
                return json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read()
            if err_body:
                try:
                    parsed = json.loads(err_body.decode("utf-8"))
                    if isinstance(parsed, dict) and ("error" in parsed or "jsonrpc" in parsed):
                        return parsed
                except Exception:
                    pass
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": f"HTTP {e.code}: {e.reason}"}
            }
        except urllib.error.URLError as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": f"Connection error: {e.reason}"}
            }
        except socket.timeout:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": f"Request timed out after {self.timeout}s"}
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": f"Internal client error: {str(e)}"}
            }

    def list_tools(self) -> List[Dict[str, Any]]:
        """Query the MCP server for available tools via standard tools/list."""
        resp = self.send_rpc("tools/list", {})
        if "error" in resp:
            err = resp["error"]
            err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise RuntimeError(f"MCP tools/list failed: {err_msg}")
        result = resp.get("result", {})
        if isinstance(result, dict) and "tools" in result:
            return result["tools"]
        if isinstance(result, list):
            return result
        return []

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool via standard tools/call and format the response as a string."""
        resp = self.send_rpc("tools/call", {"name": tool_name, "arguments": arguments})
        if "error" in resp:
            err = resp["error"]
            err_code = err.get("code", "UNKNOWN") if isinstance(err, dict) else "ERR"
            err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            return f"[MCP Error {err_code}] {err_msg}"

        result = resp.get("result")
        if result is None:
            return "[MCP Result: Empty response returned by server]"

        if isinstance(result, dict):
            # Check for standard MCP content blocks: {"content": [{"type": "text", "text": "..."}]}
            content = result.get("content")
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text" and "text" in item:
                            text_parts.append(str(item["text"]))
                        elif "text" in item:
                            text_parts.append(str(item["text"]))
                        else:
                            text_parts.append(json.dumps(item, indent=2))
                    else:
                        text_parts.append(str(item))
                if text_parts:
                    return "\n\n".join(text_parts)

            # Check if an isError flag is present
            if result.get("isError"):
                err_text = result.get("content") or result
                return f"[MCP Tool Error]: {err_text}"

            return json.dumps(result, indent=2)

        if isinstance(result, str):
            return result

        return json.dumps(result, indent=2)


class UniversalSchemaParser:
    """Parses arbitrary tool specifications into normalized schemas and dynamic Pydantic models."""

    @staticmethod
    def normalize_tool_spec(spec: Dict[str, Any], default_name: str = "") -> Dict[str, Any]:
        """Normalize a tool dictionary to guarantee 'name', 'description', and 'parameters' schema."""
        name = spec.get("name") or default_name
        description = spec.get("description", f"MCP Tool {name}")
        schema = (
            spec.get("inputSchema")
            or spec.get("parameters")
            or spec.get("schema")
            or {}
        )
        if not isinstance(schema, dict):
            schema = {}
        return {
            "name": name,
            "description": description,
            "parameters": schema
        }

    @staticmethod
    def json_schema_to_pydantic(model_name: str, schema: Dict[str, Any]) -> type:
        """Dynamically create a Pydantic model class from a JSON Schema dictionary."""
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        required_fields = set(schema.get("required", []))
        fields: Dict[str, Any] = {}

        for prop_name, prop_spec in properties.items():
            if not isinstance(prop_spec, dict):
                prop_spec = {}
            raw_type = prop_spec.get("type")
            py_type = JSON_TYPE_MAP.get(raw_type, Any)
            desc = prop_spec.get("description", "")
            has_default = "default" in prop_spec
            default_val = prop_spec.get("default")

            is_required = (prop_name in required_fields) and not has_default

            if is_required:
                fields[prop_name] = (py_type, Field(..., description=desc))
            else:
                fields[prop_name] = (Optional[py_type], Field(default=default_val, description=desc))

        # Handle empty parameter schemas gracefully
        if not fields:
            return create_model(model_name)

        return create_model(model_name, **fields)

    @classmethod
    def load_specs_from_file_or_dir(cls, path_str: str) -> List[Dict[str, Any]]:
        """Load tool specifications from a JSON file, directory of JSON files, or raw JSON string."""
        path = Path(path_str).resolve() if not path_str.strip().startswith("{") and not path_str.strip().startswith("[") else None

        if path and path.is_file():
            content = path.read_text(encoding="utf-8").strip()
            data = json.loads(content)
            return cls._parse_loaded_data(data, default_name=path.stem)

        if path and path.is_dir():
            specs = []
            for json_file in sorted(path.glob("*.json")):
                try:
                    content = json_file.read_text(encoding="utf-8").strip()
                    data = json.loads(content)
                    file_specs = cls._parse_loaded_data(data, default_name=json_file.stem)
                    specs.extend(file_specs)
                except Exception as e:
                    print(f"Warning: Failed to load tool schema from {json_file.name}: {e}", file=sys.stderr)
            return specs

        # Attempt parsing as raw JSON string
        try:
            data = json.loads(path_str)
            return cls._parse_loaded_data(data)
        except json.JSONDecodeError as e:
            raise FileNotFoundError(f"Tools info path or JSON invalid: '{path_str}' ({e})")

    @classmethod
    def _parse_loaded_data(cls, data: Any, default_name: str = "tool") -> List[Dict[str, Any]]:
        """Parse raw JSON data (list, dict with 'tools', or map) into normalized tool specs."""
        if isinstance(data, list):
            return [cls.normalize_tool_spec(item) for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            if "tools" in data and isinstance(data["tools"], list):
                return [cls.normalize_tool_spec(item) for item in data["tools"] if isinstance(item, dict)]
            if "name" in data and ("inputSchema" in data or "parameters" in data or "description" in data):
                return [cls.normalize_tool_spec(data, default_name=default_name)]
            # Map of tool_name -> tool_spec
            specs = []
            for tool_name, tool_spec in data.items():
                if isinstance(tool_spec, dict):
                    specs.append(cls.normalize_tool_spec(tool_spec, default_name=tool_name))
            return specs
        return []


def create_mcp_tool(tool_spec: Dict[str, Any], client: MCPClient) -> StructuredTool:
    """Create a LangChain StructuredTool instance from a normalized tool specification and MCP client."""
    tool_name = tool_spec["name"]
    description = tool_spec.get("description", f"MCP Tool: {tool_name}")
    parameters = tool_spec.get("parameters", {})

    model_class_name = "".join(part.capitalize() for part in tool_name.replace("-", "_").split("_")) + "Input"
    args_schema = UniversalSchemaParser.json_schema_to_pydantic(model_class_name, parameters)

    def tool_callable(**kwargs) -> str:
        # Check if user permission is required
        if get_require_permission():
            args_preview = json.dumps(kwargs)
            if len(args_preview) > 120:
                args_preview = args_preview[:120] + "..."
            perm = ask_user_permission(f"[Permission Request] Execute MCP tool '{tool_name}' ({args_preview})?")
            if not perm.allowed:
                reason_suffix = f" with reason: {perm.reason}" if perm.reason else ""
                return f"[Permission Denied] User rejected execution of MCP tool '{tool_name}'{reason_suffix}"

        return client.call_tool(tool_name, kwargs)

    tool = StructuredTool.from_function(
        func=tool_callable,
        name=tool_name,
        description=description,
        args_schema=args_schema
    )
    return tool


def load_mcp_config_file(config_path: Union[str, Path]) -> List[StructuredTool]:
    """
    Parse a standard MCP configuration JSON file and instantiate all defined MCP tools.
    Supported format:
    {
      "mcpServers": {
        "server_name": {
          "url": "https://...",
          "tools": "/path/to/tools.json" or "auto",
          "apiKey": "...",
          "headers": { ... }
        }
      }
    }
    """
    p = Path(config_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"MCP configuration file not found: {config_path}")

    config_data = json.loads(p.read_text(encoding="utf-8"))
    servers = config_data.get("mcpServers", {})
    all_tools: List[StructuredTool] = []

    for srv_name, srv_conf in servers.items():
        if not isinstance(srv_conf, dict):
            continue
        url = srv_conf.get("url")
        if not url:
            continue
        api_key = srv_conf.get("apiKey") or srv_conf.get("api_key")
        headers = srv_conf.get("headers") or {}
        tools_source = srv_conf.get("tools") or "auto"

        client = MCPClient(base_url=url, headers=headers, api_key=api_key)
        if tools_source == "auto" or not tools_source:
            specs = client.list_tools()
            normalized = [UniversalSchemaParser.normalize_tool_spec(s) for s in specs]
        else:
            normalized = UniversalSchemaParser.load_specs_from_file_or_dir(tools_source)

        for spec in normalized:
            all_tools.append(create_mcp_tool(spec, client))

    return all_tools


def build_mcp_tools(
    server_url: Optional[str] = None,
    tools_info: Optional[str] = None,
    api_key: Optional[str] = None,
    headers: Optional[Union[str, Dict[str, str]]] = None,
    config_path: Optional[str] = None
) -> List[StructuredTool]:
    """
    Universal entry point to build LangChain StructuredTools from any user-supplied MCP server.
    Accepts explicit URL + tools info, or a path to a configuration file.
    """
    if config_path:
        return load_mcp_config_file(config_path)

    if not server_url:
        return []

    # Parse headers if passed as JSON string
    parsed_headers: Dict[str, str] = {}
    if isinstance(headers, str) and headers.strip():
        try:
            parsed_headers = json.loads(headers)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string passed to --mcp-headers: {e}")
    elif isinstance(headers, dict):
        parsed_headers = headers

    client = MCPClient(base_url=server_url, headers=parsed_headers, api_key=api_key)

    # Determine tool specifications: either explicit file/dir/string or dynamic auto-discovery
    tools_source = (tools_info or "").strip()
    if not tools_source or tools_source.lower() == "auto":
        raw_specs = client.list_tools()
        specs = [UniversalSchemaParser.normalize_tool_spec(s) for s in raw_specs]
    else:
        specs = UniversalSchemaParser.load_specs_from_file_or_dir(tools_source)

    if not specs:
        raise ValueError(
            f"No tool definitions found for MCP server at '{server_url}'. "
            "Please ensure the server supports 'tools/list' or pass a valid tools schema path via --mcp-tools."
        )

    tools: List[StructuredTool] = []
    for spec in specs:
        tools.append(create_mcp_tool(spec, client))

    return tools


__all__ = [
    "MCPClient",
    "UniversalSchemaParser",
    "create_mcp_tool",
    "load_mcp_config_file",
    "build_mcp_tools",
]
