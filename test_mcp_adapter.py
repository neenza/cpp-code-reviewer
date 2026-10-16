#!/usr/bin/env python3
"""
Unit tests for Generic MCP Tool Adapter (mcp_tool_adapter.py).
Tests schema normalization, dynamic Pydantic models, JSON file/directory loading,
JSON-RPC 2.0 communication over mock HTTP server, and permission checks.
"""

import os
import sys
import json
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any

from cpp_agent_tools import (
    set_require_permission,
    get_require_permission,
    set_permission_callback,
    PermissionResult
)
from mcp_tool_adapter import (
    MCPClient,
    UniversalSchemaParser,
    create_mcp_tool,
    load_mcp_config_file,
    build_mcp_tools,
)


class MockMCPServerHandler(BaseHTTPRequestHandler):
    """Mock HTTP server handling standard MCP JSON-RPC 2.0 requests."""

    # Server state class attributes for assertions
    last_request: Dict[str, Any] = {}
    last_headers: Dict[str, str] = {}
    tools_to_list = [
        {
            "name": "custom_search",
            "description": "Search custom knowledge base",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword"},
                    "limit": {"type": "integer", "description": "Max results", "default": 5}
                },
                "required": ["query"]
            }
        }
    ]

    def log_message(self, format, *args):
        pass  # Silence HTTP server logs during tests

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        req_json = json.loads(body)
        MockMCPServerHandler.last_request = req_json
        MockMCPServerHandler.last_headers = dict(self.headers)

        method = req_json.get("method")
        req_id = req_json.get("id")
        params = req_json.get("params", {})

        if method == "tools/list":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": MockMCPServerHandler.tools_to_list}
            }
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            if tool_name == "error_tool":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": "Simulated tool execution error"}
                }
            else:
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": f"Executed {tool_name} with query='{tool_args.get('query')}'"}
                        ]
                    }
                }
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method {method} not found"}
            }

        resp_bytes = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)


class TestMCPToolAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start background HTTP server
        cls.httpd = HTTPServer(("127.0.0.1", 0), MockMCPServerHandler)
        cls.port = cls.httpd.server_port
        cls.server_url = f"http://127.0.0.1:{cls.port}/mcp"
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        set_permission_callback(None)
        set_require_permission(False)

    def tearDown(self):
        set_permission_callback(None)
        set_require_permission(False)

    def test_json_schema_to_pydantic_all_primitive_and_complex_types(self):
        schema = {
            "type": "object",
            "properties": {
                "str_val": {"type": "string", "description": "A string"},
                "int_val": {"type": "integer", "description": "An integer", "default": 42},
                "num_val": {"type": "number", "description": "A float number"},
                "bool_val": {"type": "boolean", "description": "A boolean flag"},
                "list_val": {"type": "array", "description": "A list of items"},
                "dict_val": {"type": "object", "description": "A dict of values"},
            },
            "required": ["str_val", "bool_val"]
        }
        model = UniversalSchemaParser.json_schema_to_pydantic("CustomInput", schema)
        fields = model.model_fields
        self.assertIn("str_val", fields)
        self.assertIn("int_val", fields)
        self.assertIn("num_val", fields)
        self.assertIn("bool_val", fields)
        self.assertIn("list_val", fields)
        self.assertIn("dict_val", fields)

        # Validate instantiating model
        instance = model(str_val="hello", bool_val=True)
        self.assertEqual(instance.str_val, "hello")
        self.assertEqual(instance.int_val, 42)
        self.assertEqual(instance.bool_val, True)
        self.assertIsNone(instance.num_val)

    def test_empty_parameter_schema(self):
        empty_schema = {"type": "object", "properties": {}}
        model = UniversalSchemaParser.json_schema_to_pydantic("EmptyInput", empty_schema)
        instance = model()
        self.assertIsNotNone(instance)

    def test_load_specs_from_single_file_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "tools.json"
            fpath.write_text(json.dumps([
                {
                    "name": "tool_a",
                    "description": "Tool A",
                    "parameters": {"properties": {"x": {"type": "string"}}}
                },
                {
                    "name": "tool_b",
                    "description": "Tool B",
                    "inputSchema": {"properties": {"y": {"type": "integer"}}}
                }
            ]))

            specs = UniversalSchemaParser.load_specs_from_file_or_dir(str(fpath))
            self.assertEqual(len(specs), 2)
            self.assertEqual(specs[0]["name"], "tool_a")
            self.assertEqual(specs[1]["name"], "tool_b")

    def test_load_specs_from_single_file_dict_tools(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "mcp_tools.json"
            fpath.write_text(json.dumps({
                "tools": [
                    {"name": "fetch_data", "description": "Fetch data", "inputSchema": {}}
                ]
            }))

            specs = UniversalSchemaParser.load_specs_from_file_or_dir(str(fpath))
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0]["name"], "fetch_data")

    def test_load_specs_from_directory_of_schemas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirpath = Path(tmpdir)
            (dirpath / "query.json").write_text(json.dumps({
                "name": "query",
                "description": "Query database",
                "parameters": {"properties": {"sql": {"type": "string"}}}
            }))
            (dirpath / "mutate.json").write_text(json.dumps({
                "name": "mutate",
                "description": "Mutate database",
                "parameters": {"properties": {"cmd": {"type": "string"}}}
            }))

            specs = UniversalSchemaParser.load_specs_from_file_or_dir(str(dirpath))
            self.assertEqual(len(specs), 2)
            names = {s["name"] for s in specs}
            self.assertEqual(names, {"query", "mutate"})

    def test_load_specs_from_raw_json_string(self):
        raw_json = json.dumps([
            {"name": "inline_tool", "description": "Inline tool", "parameters": {}}
        ])
        specs = UniversalSchemaParser.load_specs_from_file_or_dir(raw_json)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["name"], "inline_tool")

    def test_mcp_client_tools_list_and_auth_headers(self):
        client = MCPClient(
            base_url=self.server_url,
            headers={"X-Custom": "val-123"},
            api_key="secret-key-abc"
        )
        tools = client.list_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "custom_search")
        self.assertEqual(MockMCPServerHandler.last_headers.get("X-Api-Key"), "secret-key-abc")
        self.assertEqual(MockMCPServerHandler.last_headers.get("X-Custom"), "val-123")
        self.assertEqual(MockMCPServerHandler.last_headers.get("Authorization"), "Bearer secret-key-abc")

    def test_mcp_client_tools_call_success_and_error(self):
        client = MCPClient(base_url=self.server_url)

        # Success call
        res = client.call_tool("custom_search", {"query": "hello world"})
        self.assertIn("Executed custom_search with query='hello world'", res)

        # Error call
        err_res = client.call_tool("error_tool", {})
        self.assertIn("[MCP Error -32602]", err_res)
        self.assertIn("Simulated tool execution error", err_res)

    def test_create_mcp_tool_execution_with_permission_flow(self):
        client = MCPClient(base_url=self.server_url)
        spec = {
            "name": "custom_search",
            "description": "Search custom knowledge base",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
        tool = create_mcp_tool(spec, client)

        # Test 1: Auto-approve / permissions disabled
        set_require_permission(False)
        out = tool.invoke({"query": "antigravity"})
        self.assertIn("Executed custom_search with query='antigravity'", out)

        # Test 2: Interactive permissions enabled & rejected with reason
        set_require_permission(True)
        set_permission_callback(lambda prompt: PermissionResult(False, "Query contains forbidden keyword"))
        out_rejected = tool.invoke({"query": "forbidden"})
        self.assertIn("[Permission Denied]", out_rejected)
        self.assertIn("with reason: Query contains forbidden keyword", out_rejected)

        # Test 3: Interactive permissions enabled & approved
        set_permission_callback(lambda prompt: PermissionResult(True, ""))
        out_approved = tool.invoke({"query": "approved_query"})
        self.assertIn("Executed custom_search with query='approved_query'", out_approved)

    def test_build_mcp_tools_with_auto_discovery(self):
        tools = build_mcp_tools(server_url=self.server_url, tools_info="auto")
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "custom_search")
        res = tools[0].invoke({"query": "auto_discovered"})
        self.assertIn("Executed custom_search with query='auto_discovered'", res)

    def test_load_mcp_config_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = Path(tmpdir) / "mcp_config.json"
            cfg_file.write_text(json.dumps({
                "mcpServers": {
                    "mock_service": {
                        "url": self.server_url,
                        "apiKey": "test-key-123",
                        "tools": "auto"
                    }
                }
            }))

            tools = load_mcp_config_file(str(cfg_file))
            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0].name, "custom_search")

    def test_coding_assistant_graph_with_mcp_tool_execution(self):
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
        from coding_assistant_agent import build_coding_assistant_graph
        from cpp_agent_tools import CODING_ASSISTANT_TOOLS

        tools = build_mcp_tools(server_url=self.server_url, tools_info="auto")
        all_tools = list(CODING_ASSISTANT_TOOLS) + list(tools)

        class ScriptedLLMWithMCP:
            def __init__(self):
                self.calls = 0
            def bind_tools(self, tools):
                self.bound_tools = tools
                return self
            def invoke(self, messages):
                self.calls += 1
                if self.calls == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "custom_search",
                            "args": {"query": "graph_pipeline_test"},
                            "id": "call_mcp_1"
                        }]
                    )
                else:
                    return AIMessage(content="MCP tool completed search successfully.")

        mock_llm = ScriptedLLMWithMCP()
        app = build_coding_assistant_graph(llm=mock_llm, tools=all_tools)

        initial_state = {
            "messages": [HumanMessage(content="Search for pipeline test")]
        }
        res = app.invoke(initial_state, {"recursion_limit": 10})
        messages = res["messages"]

        # Verify that ToolNode executed custom_search and injected ToolMessage
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0].name, "custom_search")
        self.assertIn("Executed custom_search with query='graph_pipeline_test'", tool_msgs[0].content)

        # Verify final assistant message
        final_msg = messages[-1]
        self.assertIn("MCP tool completed search successfully.", final_msg.content)

    def test_parse_args_mcp_options(self):
        from unittest.mock import patch
        from coding_assistant_agent import parse_args

        test_args = [
            "coding_assistant_agent.py",
            "/path/to/repo",
            "--mcp-server-url", "http://localhost:8080/mcp",
            "--mcp-tools", "tools_spec.json",
            "--mcp-api-key", "my_key_999",
            "--mcp-headers", '{"X-Tenant": "dev"}',
            "--mcp-config", "mcp_config.json"
        ]
        with patch.object(sys, "argv", test_args):
            parsed = parse_args()
            self.assertEqual(parsed.project_dir, "/path/to/repo")
            self.assertEqual(parsed.mcp_server_url, "http://localhost:8080/mcp")
            self.assertEqual(parsed.mcp_tools, "tools_spec.json")
            self.assertEqual(parsed.mcp_api_key, "my_key_999")
            self.assertEqual(parsed.mcp_headers, '{"X-Tenant": "dev"}')
            self.assertEqual(parsed.mcp_config, "mcp_config.json")


if __name__ == "__main__":
    unittest.main()
