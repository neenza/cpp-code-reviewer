import os
import unittest
from pathlib import Path
import tempfile

# Shared Tools
from cpp_agent_tools import (
    clangd_query,
    ripgrep_search,
    read_project_file,
    set_active_project_dir,
    COMMON_CPP_TOOLS
)

# Review Agent
from code_review_agent import (
    record_finding,
    build_repo_review_orchestrator,
    _RECORDED_FINDINGS
)

# Explainer Agent
from code_explainer_agent import build_explainer_graph

# Documenter Agent
from code_documenter_agent import (
    init_documentation_file,
    append_documentation_section,
    read_current_documentation_toc,
    trim_messages_to_budget,
    build_codebase_documenter_graph
)

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage


class TestCodeReviewExplainerAndDocumenterTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_dir = Path(__file__).parent / "sample_project"
        set_active_project_dir(cls.sample_dir)

    def test_read_cmakelists(self):
        content = read_project_file.invoke({"file_path": "CMakeLists.txt"})
        self.assertIn("SampleOrderSystem", content)
        self.assertIn("CMAKE_CXX_STANDARD 17", content)

    def test_ripgrep_search(self):
        res = ripgrep_search.invoke({"pattern": "strcpy", "path_filter": "src"})
        self.assertIn("session_manager.cpp", res)

        res_mutex = ripgrep_search.invoke({"pattern": "std::shared_mutex", "path_filter": "include"})
        self.assertIn("order_repository.h", res_mutex)

    def test_clangd_query_search(self):
        res = clangd_query.invoke({"command": "search", "symbol_or_query": "OrderRepository"})
        self.assertIn("OrderRepository", res)

    def test_clangd_query_show(self):
        res = clangd_query.invoke({"command": "show", "symbol_or_query": "OrderRepository"})
        self.assertIn("class OrderRepository", res)

    def test_clangd_query_interface(self):
        res = clangd_query.invoke({"command": "interface", "symbol_or_query": "IPaymentGateway"})
        self.assertIn("process", res)

    def test_clangd_query_hierarchy(self):
        res = clangd_query.invoke({"command": "hierarchy", "symbol_or_query": "StripeGateway"})
        self.assertIn("IPaymentGateway", res)

    def test_record_finding(self):
        res = record_finding.invoke({
            "category": "critical_flaw",
            "title": "Buffer overflow in SessionManager",
            "details": "strcpy without bounds check",
            "files_and_lines": "src/session_manager.cpp:18",
            "recommended_fix": "Use std::string or snprintf"
        })
        self.assertIn("CRITICAL_FLAW", res)
        self.assertTrue(any(f["title"] == "Buffer overflow in SessionManager" for f in _RECORDED_FINDINGS))

    def test_review_orchestrator_compilation(self):
        class DummyLLM:
            def bind_tools(self, tools):
                return self
            def invoke(self, messages):
                return AIMessage(content="Module review finished.")

        graph = build_repo_review_orchestrator(llm=DummyLLM())
        self.assertIsNotNone(graph)

    def test_explainer_graph_compilation(self):
        class DummyLLM:
            def bind_tools(self, tools):
                return self
            def invoke(self, messages):
                return AIMessage(content="Here is how the architecture works.")

        graph = build_explainer_graph(llm=DummyLLM(), tools=COMMON_CPP_TOOLS)
        self.assertIsNotNone(graph)

    def test_documenter_incremental_writer(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        init_documentation_file(tmp_path, "SampleOrderSystem")
        res = append_documentation_section.invoke({
            "section_title": "Order Repository Architecture",
            "markdown_content": "Thread-safe concurrent order store using `std::shared_mutex`.",
            "level": 2
        })
        self.assertIn("Successfully appended", res)

        toc = read_current_documentation_toc.invoke({})
        self.assertIn("Order Repository Architecture", toc)

        with open(tmp_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("## 1. Order Repository Architecture", content)
        self.assertIn("Thread-safe concurrent order store", content)

        if tmp_path.exists():
            tmp_path.unlink()

    def test_context_trimmer_32k_budget(self):
        # Create a series of messages exceeding small budget to verify trimming
        sys_msg = SystemMessage(content="You are a documenter.")
        human_msg = HumanMessage(content="Document module include.")
        # Large tool content
        large_tool_msg = ToolMessage(content="A" * 12000, tool_call_id="call_1", name="read_project_file")
        ai_msg = AIMessage(content="Continuing exploration", tool_calls=[{"id": "call_2", "name": "clangd_query", "args": {}}])
        recent_tool = ToolMessage(content="class OrderRepository {}", tool_call_id="call_2", name="clangd_query")

        messages = [sys_msg, human_msg, large_tool_msg, ai_msg, recent_tool]
        # Trim to 2000 tokens
        trimmed = trim_messages_to_budget(messages, max_tokens=2000, reserve_tokens=200)
        self.assertEqual(trimmed[0].content, "You are a documenter.")
        self.assertEqual(trimmed[1].content, "Document module include.")
        self.assertTrue(len(trimmed) >= 2)

    def test_documenter_graph_compilation(self):
        class DummyLLM:
            def bind_tools(self, tools):
                return self
            def invoke(self, messages):
                return AIMessage(content="Documentation completed.")

        graph = build_codebase_documenter_graph(llm=DummyLLM(), max_context_tokens=32000)
        self.assertIsNotNone(graph)

    def test_documenter_target_dirs_filtering(self):
        from code_documenter_agent import doc_discover_and_plan_node
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        state = {
            "project_dir": str(self.sample_dir),
            "output_file": str(tmp_path),
            "target_dirs": ["include"],
            "all_files": [],
            "modules": [],
            "module_files_map": {},
            "current_module_index": 0,
            "sections_count": 0,
            "max_context_tokens": 32000
        }
        res = doc_discover_and_plan_node(state)
        # Should only queue 'include', not 'src'
        self.assertEqual(res["modules"], ["include"])
        # But all_files still has the full project files for tool exploration
        self.assertTrue(any("src/" in f for f in res["all_files"]))

        if tmp_path.exists():
            tmp_path.unlink()

    def test_documenter_ignore_dirs(self):
        from code_documenter_agent import doc_discover_and_plan_node
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        state = {
            "project_dir": str(self.sample_dir),
            "output_file": str(tmp_path),
            "target_dirs": None,
            "ignore_dirs": ["include"],
            "all_files": [],
            "modules": [],
            "module_files_map": {},
            "current_module_index": 0,
            "sections_count": 0,
            "max_context_tokens": 32000
        }
        res = doc_discover_and_plan_node(state)
        # 'include' should be completely ignored
        self.assertNotIn("include", res["modules"])
        self.assertIn("src", res["modules"])

        if tmp_path.exists():
            tmp_path.unlink()

    def test_documenter_entry_point_topological_sort(self):
        from code_documenter_agent import doc_discover_and_plan_node
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        state = {
            "project_dir": str(self.sample_dir),
            "output_file": str(tmp_path),
            "target_dirs": None,
            "ignore_dirs": None,
            "all_files": [],
            "modules": [],
            "module_files_map": {},
            "current_module_index": 0,
            "sections_count": 0,
            "max_context_tokens": 32000
        }
        res = doc_discover_and_plan_node(state)
        # sample_project has main.cpp in 'src', so 'src' must be the starting point (index 0)
        self.assertEqual(res["modules"][0], "src")

        if tmp_path.exists():
            tmp_path.unlink()

    def test_automatic_sequential_section_numbering(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        init_documentation_file(tmp_path, "TestSeq")
        # Pass out of sequence or prefixed titles with substantive markdown content (>30 chars)
        append_documentation_section.invoke({
            "section_title": "Section 9: Core Engine",
            "markdown_content": "Detailed core engine architectural logic and processing workflow."
        })
        append_documentation_section.invoke({
            "section_title": "3. Data Storage",
            "markdown_content": "Thread-safe data storage repository mechanisms and indexing logic."
        })
        append_documentation_section.invoke({
            "section_title": "Networking API",
            "markdown_content": "Network protocol handling and asynchronous endpoint management."
        })

        with open(tmp_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should be strictly numbered 1, 2, 3
        self.assertIn("## 1. Core Engine", content)
        self.assertIn("## 2. Data Storage", content)
        self.assertIn("## 3. Networking API", content)

        if tmp_path.exists():
            tmp_path.unlink()

    def test_document_module_node_guaranteed_append(self):
        import code_documenter_agent
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        code_documenter_agent.init_documentation_file(tmp_path, "TestModuleAppend")

        # Dummy LLM that returns text without calling append_documentation_section tool
        class DirectTextLLM:
            def bind_tools(self, tools):
                return self
            def invoke(self, messages):
                return AIMessage(content="### Functional Explanation for src\nThe `src` module contains main.cpp which orchestrates the order processing loop and initializes repository storage.")

        node_fn = code_documenter_agent.document_module_node_factory(llm=DirectTextLLM(), max_context_tokens=32000, module_max_steps=5)
        state = {
            "project_dir": str(self.sample_dir),
            "output_file": str(tmp_path),
            "target_dirs": None,
            "ignore_dirs": None,
            "all_files": ["src/main.cpp"],
            "modules": ["src"],
            "module_files_map": {"src": ["src/main.cpp"]},
            "current_module_index": 0,
            "sections_count": 0,
            "max_context_tokens": 32000
        }

        res = node_fn(state)
        self.assertEqual(res["current_module_index"], 1)
        self.assertGreater(len(code_documenter_agent._DOCUMENTED_SECTIONS), 0)

        with open(tmp_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that the section was saved automatically even though the model didn't call the tool
        self.assertIn("Src - Functional Architecture & Implementation", content)
        self.assertIn("main.cpp which orchestrates the order processing loop", content)

        if tmp_path.exists():
            tmp_path.unlink()

    def test_parse_interface_methods(self):
        from cpp_agent_tools import parse_interface_methods
        interface_sample = """
class order_system::OrderRepository - include/order_repository.h:17:7

Public Interface:

OrderRepository() = default

~OrderRepository() = default

OrderRepository(const OrderRepository&) = delete
  Non-copyable, movable

OrderRepository& operator=(const OrderRepository&) = delete

bool add_order(const Order& order)

std::optional<Order> get_order(const std::string& order_id) const

size_t count() const noexcept
"""
        parsed = parse_interface_methods(interface_sample)
        self.assertGreater(len(parsed), 0)

        # Check trivial methods
        trivial_syms = [p["full_symbol"] for p in parsed if p["is_trivial"]]
        self.assertTrue(any("OrderRepository" in s for s in trivial_syms))
        self.assertTrue(any("operator=" in s for s in trivial_syms))

        # Check non-trivial methods
        non_trivial = [p["full_symbol"] for p in parsed if not p["is_trivial"]]
        self.assertIn("OrderRepository::add_order", non_trivial)
        self.assertIn("OrderRepository::get_order", non_trivial)
        self.assertIn("OrderRepository::count", non_trivial)

    def test_discover_module_functions(self):
        from cpp_agent_tools import discover_module_functions
        src_files = ["src/order_repository.cpp", "src/session_manager.cpp"]
        funcs = discover_module_functions(self.sample_dir, src_files)
        self.assertIn("OrderRepository::add_order", funcs)
        self.assertIn("SessionManager::create_session", funcs)
        self.assertIn("SessionManager::cleanup_all", funcs)

    def test_extract_touched_files(self):
        from cpp_agent_tools import extract_touched_files
        sample_output = """
Found method 'order_system::OrderRepository::add_order'
From include/order_repository.h:28:10 (declaration)
From src/order_repository.cpp:6:23 (definition)
"""
        project_files = ["include/order_repository.h", "src/order_repository.cpp", "src/main.cpp"]
        touched = extract_touched_files(sample_output, project_files)
        self.assertIn("include/order_repository.h", touched)
        self.assertIn("src/order_repository.cpp", touched)
        self.assertNotIn("src/main.cpp", touched)

    def test_route_module_reviewer_enforces_pending_audit(self):
        from code_review_agent import route_module_reviewer, enforce_audit_node
        # State where agent emitted NO tool calls, but functions remain unaudited
        state = {
            "messages": [AIMessage(content="I have reviewed the interface. Finished.")],
            "module_name": "src",
            "user_prompt": "",
            "target_files": ["src/session_manager.cpp"],
            "target_classes": ["SessionManager"],
            "interfaced_classes": ["SessionManager"],
            "discovered_functions": ["SessionManager::create_session", "SessionManager::cleanup_all"],
            "audited_functions": [],  # 0 audited
            "audited_files": [],
            "nudge_count": 0,
            "findings_at_start": 0
        }
        route = route_module_reviewer(state)
        self.assertEqual(route, "enforce_audit")

        # Now test enforce_audit_node generates feedback and increments nudge_count
        enforce_update = enforce_audit_node(state)
        self.assertEqual(enforce_update["nudge_count"], 1)
        feedback_content = enforce_update["messages"][0].content
        self.assertIn("AUDIT INCOMPLETE FOR MODULE 'src'", feedback_content)
        self.assertIn("SessionManager::create_session", feedback_content)

    def test_route_module_reviewer_allows_completion_when_all_audited(self):
        from code_review_agent import route_module_reviewer
        from langgraph.graph import END
        state = {
            "messages": [AIMessage(content="Everything audited.")],
            "module_name": "src",
            "user_prompt": "",
            "target_files": ["src/session_manager.cpp"],
            "target_classes": ["SessionManager"],
            "interfaced_classes": ["SessionManager"],
            "discovered_functions": ["SessionManager::create_session"],
            "audited_functions": ["SessionManager::create_session"],
            "audited_files": ["src/session_manager.cpp"],
            "nudge_count": 0,
            "findings_at_start": 0
        }
        route = route_module_reviewer(state)
        self.assertEqual(route, END)

    def test_audit_tools_node_updates_ledger(self):
        from code_review_agent import audit_tools_node
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "clangd_query",
                        "args": {"command": "interface", "symbol_or_query": "OrderRepository"},
                        "id": "tc_1"
                    }]
                )
            ],
            "module_name": "include",
            "user_prompt": "",
            "target_files": ["include/order_repository.h"],
            "target_classes": ["OrderRepository"],
            "interfaced_classes": [],
            "discovered_functions": [],
            "audited_functions": [],
            "audited_files": [],
            "nudge_count": 0,
            "findings_at_start": 0
        }
        res = audit_tools_node(state)
        self.assertIn("OrderRepository", res["interfaced_classes"])
        self.assertTrue(any("OrderRepository::add_order" in f for f in res["discovered_functions"]))
        self.assertTrue(any("OrderRepository::count" in f for f in res["discovered_functions"]))
        self.assertIn("include/order_repository.h", res["audited_files"])

    def test_end_to_end_module_reviewer_flow_with_audit_enforcement(self):
        from code_review_agent import build_module_reviewer

        class ScriptedLLM:
            def __init__(self):
                self.calls = 0

            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                self.calls += 1
                # Turn 1: Query interface
                if self.calls == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "clangd_query",
                            "args": {"command": "interface", "symbol_or_query": "SessionManager"},
                            "id": "tc_interface"
                        }]
                    )
                # Turn 2: Attempt premature conclusion without tool calls
                elif self.calls == 2:
                    return AIMessage(content="Interface reviewed. Moving on.")
                # Turn 3: Received enforce_audit nudge! Inspect create_session & record finding
                elif self.calls == 3:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "clangd_query",
                                "args": {"command": "show", "symbol_or_query": "SessionManager::create_session"},
                                "id": "tc_show"
                            },
                            {
                                "name": "record_finding",
                                "args": {
                                    "category": "critical_flaw",
                                    "title": "Buffer Overflow in SessionManager::create_session",
                                    "details": "strcpy without bounds checking",
                                    "files_and_lines": "src/session_manager.cpp:18"
                                },
                                "id": "tc_finding"
                            }
                        ]
                    )
                # Turn 4+: Mark remaining functions
                else:
                    return AIMessage(content="All code examined.")

        app = build_module_reviewer(llm=ScriptedLLM(), max_context_tokens=32000)
        initial_state = {
            "messages": [HumanMessage(content="Audit module src")],
            "module_name": "src",
            "user_prompt": "",
            "target_files": ["src/session_manager.cpp"],
            "target_classes": ["SessionManager"],
            "interfaced_classes": [],
            "discovered_functions": [],
            "audited_functions": [],
            "audited_files": [],
            "nudge_count": 0,
            "findings_at_start": 0,
            "max_nudges": 2
        }
        res = app.invoke(initial_state, {"recursion_limit": 30})
        self.assertGreater(res["nudge_count"], 0)
        self.assertIn("SessionManager", res["interfaced_classes"])
        self.assertTrue(any("create_session" in f for f in res["audited_functions"]))
        # Verify finding was recorded
        global _RECORDED_FINDINGS
        self.assertTrue(any("Buffer Overflow in SessionManager::create_session" in f["title"] for f in _RECORDED_FINDINGS))

    def test_manage_context_with_summarization_includes_coverage_ledger(self):
        from code_review_agent import manage_context_with_summarization
        from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

        sys_msg = SystemMessage(content="You are a code auditor.")
        human_msg = HumanMessage(content="Review module src.")
        large_tool = ToolMessage(content="class SessionManager { ... }" + "A" * 15000, tool_call_id="call_1", name="clangd_query")
        ai_msg = AIMessage(content="I inspected SessionManager.")
        recent_tool = ToolMessage(content="void cleanup_all();", tool_call_id="call_2", name="clangd_query")

        messages = [sys_msg, human_msg, large_tool, ai_msg, recent_tool]
        audit_state = {
            "module_name": "src",
            "target_classes": ["SessionManager", "PaymentProcessor"],
            "interfaced_classes": ["SessionManager"],
            "discovered_functions": [
                "SessionManager::create_session",
                "SessionManager::cleanup_all",
                "PaymentProcessor::process"
            ],
            "audited_functions": ["SessionManager::create_session"],
            "target_files": ["src/session_manager.cpp", "src/payment_processor.cpp"],
            "audited_files": ["src/session_manager.cpp"]
        }

        trimmed = manage_context_with_summarization(
            messages,
            max_tokens=4000,
            reserve_tokens=500,
            audit_state=audit_state
        )

        # The summary message is placed at index 2 (after sys_msg and human_msg)
        summary_content = trimmed[2].content
        self.assertIn("Audit Progress Ledger for Module 'src'", summary_content)
        self.assertIn("ALREADY COVERED (DO NOT RE-AUDIT)", summary_content)
        self.assertIn("SessionManager::create_session", summary_content)
        self.assertIn("REMAINING TO BE AUDITED (PRIORITIZE)", summary_content)
        self.assertIn("SessionManager::cleanup_all", summary_content)
        self.assertIn("ALREADY INTERFACED (DO NOT RE-QUERY)", summary_content)
        self.assertIn("SessionManager", summary_content)
        self.assertIn("REMAINING CLASSES TO INTERFACE", summary_content)
        self.assertIn("PaymentProcessor", summary_content)
        self.assertIn("REMAINING UNINSPECTED FILES", summary_content)
        self.assertIn("src/payment_processor.cpp", summary_content)

    def test_partition_messages_safely_keeps_all_trailing_tool_messages_intact(self):
        from code_review_agent import partition_messages_safely
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

        # Scenario: agent emitted 4 tool calls; all 4 ToolMessages are at the tail of messages
        m0 = HumanMessage(content="Audit the module")
        m1 = AIMessage(content="Older note")
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "clangd_query", "args": {"command": "show", "symbol_or_query": "sym1"}, "id": "tc1"},
                {"name": "clangd_query", "args": {"command": "show", "symbol_or_query": "sym2"}, "id": "tc2"},
                {"name": "clangd_query", "args": {"command": "show", "symbol_or_query": "sym3"}, "id": "tc3"},
                {"name": "clangd_query", "args": {"command": "show", "symbol_or_query": "sym4"}, "id": "tc4"},
            ]
        )
        t1 = ToolMessage(content="void sym1() { /* body */ }", tool_call_id="tc1", name="clangd_query")
        t2 = ToolMessage(content="void sym2() { /* body */ }", tool_call_id="tc2", name="clangd_query")
        t3 = ToolMessage(content="void sym3() { /* body */ }", tool_call_id="tc3", name="clangd_query")
        t4 = ToolMessage(content="void sym4() { /* body */ }", tool_call_id="tc4", name="clangd_query")

        messages = [m0, m1, ai_msg, t1, t2, t3, t4]

        older, recent = partition_messages_safely(messages, target_recent_count=2)
        # Recent MUST contain ai_msg and all 4 ToolMessages, NEVER empty
        self.assertEqual(len(recent), 5)
        self.assertIs(recent[0], ai_msg)
        self.assertIs(recent[1], t1)
        self.assertIs(recent[2], t2)
        self.assertIs(recent[3], t3)
        self.assertIs(recent[4], t4)
        # Older should contain only m0, m1
        self.assertEqual(len(older), 2)
        self.assertIs(older[0], m0)
        self.assertIs(older[1], m1)

    def test_in_flight_functions_not_marked_as_covered_during_summarization(self):
        from code_review_agent import manage_context_with_summarization
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

        # Scenario: LLM just requested 3 symbols, and the tools returned large outputs crossing context limit
        sys_msg = SystemMessage(content="System prompt")
        user_msg = HumanMessage(content="Review module")
        older_ai = AIMessage(content="I previously inspected the class interface and found methods.")
        ai_req = AIMessage(
            content="",
            tool_calls=[
                {"name": "clangd_query", "args": {"command": "show", "symbol_or_query": "SessionManager::create_session"}, "id": "c1"},
                {"name": "clangd_query", "args": {"command": "show", "symbol_or_query": "SessionManager::invalidate_session"}, "id": "c2"},
                {"name": "clangd_query", "args": {"command": "show", "symbol_or_query": "SessionManager::get_session"}, "id": "c3"},
            ]
        )
        # Large bodies that push tokens past threshold
        t1 = ToolMessage(content="Session create_session() {\n" + "    // code\n" * 200 + "}", tool_call_id="c1", name="clangd_query")
        t2 = ToolMessage(content="void invalidate_session() {\n" + "    // code\n" * 200 + "}", tool_call_id="c2", name="clangd_query")
        t3 = ToolMessage(content="Session* get_session() {\n" + "    // code\n" * 200 + "}", tool_call_id="c3", name="clangd_query")

        messages = [sys_msg, user_msg, older_ai, ai_req, t1, t2, t3]

        audit_state = {
            "module_name": "src",
            "target_classes": ["SessionManager"],
            "interfaced_classes": ["SessionManager"],
            "discovered_functions": [
                "SessionManager::create_session",
                "SessionManager::invalidate_session",
                "SessionManager::get_session",
                "SessionManager::cleanup_all"
            ],
            "audited_functions": [],  # NOT audited yet!
            "in_flight_functions": [
                "SessionManager::create_session",
                "SessionManager::invalidate_session",
                "SessionManager::get_session"
            ],
            "target_files": ["src/session_manager.cpp"],
            "audited_files": []
        }

        trimmed = manage_context_with_summarization(
            messages,
            max_tokens=2500,
            reserve_tokens=300,
            summarize_threshold=1000,
            audit_state=audit_state
        )

        summary_msg = trimmed[2]
        content = summary_msg.content

        # 1. Must NOT be marked as ALREADY COVERED (DO NOT RE-AUDIT)
        self.assertNotIn("ALREADY COVERED (DO NOT RE-AUDIT): `SessionManager::create_session`", content)
        self.assertNotIn("ALREADY COVERED (DO NOT RE-AUDIT): `SessionManager::invalidate_session`", content)
        self.assertNotIn("ALREADY COVERED (DO NOT RE-AUDIT): `SessionManager::get_session`", content)

        # 2. Must be explicitly flagged as CURRENTLY UNDER ACTIVE REVIEW
        self.assertIn("CURRENTLY UNDER ACTIVE REVIEW", content)
        self.assertIn("SessionManager::create_session", content)
        self.assertIn("SessionManager::invalidate_session", content)
        self.assertIn("SessionManager::get_session", content)

        # 3. Remaining unqueried functions still prioritized
        self.assertIn("REMAINING TO BE AUDITED (PRIORITIZE)", content)
        self.assertIn("SessionManager::cleanup_all", content)

        # 4. Active tool messages must be preserved intact in recent_active_turns
        self.assertIn(t1, trimmed)
        self.assertIn(t2, trimmed)
        self.assertIn(t3, trimmed)
        self.assertIn(ai_req, trimmed)

    def test_documenter_module_routing_with_enforcement(self):
        from code_documenter_agent import build_module_documenter_runner, init_documentation_file
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            test_doc = Path(tmpdir) / "TEST_DOC.md"
            init_documentation_file(test_doc, "TestProject")

            class ScriptedDocumenterLLM:
                def __init__(self):
                    self.calls = 0

                def bind_tools(self, tools):
                    return self

                def invoke(self, messages):
                    self.calls += 1
                    # Turn 1: Query class interface
                    if self.calls == 1:
                        return AIMessage(
                            content="",
                            tool_calls=[{
                                "name": "clangd_query",
                                "args": {"command": "interface", "symbol_or_query": "SessionManager"},
                                "id": "tc_iface"
                            }]
                        )
                    # Turn 2: Premature exit without tool calls
                    elif self.calls == 2:
                        return AIMessage(content="SessionManager looks nice. Concluding module.")
                    # Turn 3: Received enforce_documentation nudge! Inspect method and append section
                    elif self.calls == 3:
                        return AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "clangd_query",
                                    "args": {"command": "show", "symbol_or_query": "SessionManager::create_session"},
                                    "id": "tc_show"
                                },
                                {
                                    "name": "append_documentation_section",
                                    "args": {
                                        "section_title": "Session Management Subsystem",
                                        "markdown_content": "### Architecture\nSessionManager handles sessions.\n```mermaid\ngraph TD\n  A --> B\n```",
                                        "level": 2
                                    },
                                    "id": "tc_append"
                                }
                            ]
                        )
                    # Turn 4: Final response
                    else:
                        return AIMessage(content="Documentation is complete.")

            app = build_module_documenter_runner(llm=ScriptedDocumenterLLM(), max_context_tokens=32000)
            initial_state = {
                "messages": [HumanMessage(content="Document module src")],
                "module_name": "src",
                "user_prompt": "",
                "target_files": ["src/session_manager.cpp"],
                "target_classes": ["SessionManager"],
                "interfaced_classes": [],
                "discovered_functions": [],
                "documented_functions": [],
                "in_flight_functions": [],
                "documented_files": [],
                "nudge_count": 0,
                "max_nudges": 2,
                "append_called": False,
                "append_count": 0,
                "uncommitted_explorations": 0,
                "reminder_count": 0
            }

            res = app.invoke(initial_state, {"recursion_limit": 30})
            self.assertGreater(res["nudge_count"], 0)
            self.assertIn("SessionManager", res["interfaced_classes"])
            self.assertTrue(any("create_session" in f for f in res["documented_functions"]))
            self.assertTrue(res["append_called"])
            self.assertGreater(res["append_count"], 0)
            self.assertTrue(test_doc.exists())
            doc_text = test_doc.read_text(encoding="utf-8")
            self.assertIn("Session Management Subsystem", doc_text)

    def test_documenter_manage_context_with_summarization_ledger(self):
        from code_documenter_agent import manage_context_with_summarization
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

        sys_msg = SystemMessage(content="System prompt")
        user_msg = HumanMessage(content="Document module src")
        older_ai = AIMessage(content="Older exploration turn notes.")
        ai_req = AIMessage(
            content="",
            tool_calls=[
                {"name": "clangd_query", "args": {"command": "show", "symbol_or_query": "OrderRepository::add_order"}, "id": "call_add"}
            ]
        )
        large_tool = ToolMessage(
            content="void add_order() {\n" + "    // implementation code\n" * 300 + "}",
            tool_call_id="call_add",
            name="clangd_query"
        )

        messages = [sys_msg, user_msg, older_ai, ai_req, large_tool]

        audit_state = {
            "module_name": "src",
            "target_classes": ["OrderRepository", "PaymentProcessor"],
            "interfaced_classes": ["OrderRepository"],
            "discovered_functions": [
                "OrderRepository::add_order",
                "OrderRepository::count",
                "PaymentProcessor::process"
            ],
            "documented_functions": [],
            "in_flight_functions": ["OrderRepository::add_order"],
            "target_files": ["src/order_repository.cpp", "src/payment_processor.cpp"],
            "documented_files": ["src/order_repository.cpp"]
        }

        trimmed = manage_context_with_summarization(
            messages,
            max_tokens=2500,
            reserve_tokens=300,
            summarize_threshold=800,
            audit_state=audit_state
        )

        summary_msg = trimmed[2]
        content = summary_msg.content

        self.assertIn("Documentation Progress Ledger for Module 'src'", content)
        self.assertIn("CURRENTLY UNDER ACTIVE REVIEW", content)
        self.assertIn("OrderRepository::add_order", content)
        self.assertIn("REMAINING TO BE DOCUMENTED (PRIORITIZE)", content)
        self.assertIn("OrderRepository::count", content)
        self.assertIn("PaymentProcessor::process", content)
        self.assertIn("ALREADY INTERFACED (DO NOT RE-QUERY)", content)
        self.assertIn("OrderRepository", content)
        self.assertIn("REMAINING CLASSES TO INTERFACE", content)
        self.assertIn("PaymentProcessor", content)
        self.assertIn("REMAINING UNINSPECTED FILES", content)
        self.assertIn("src/payment_processor.cpp", content)

        # Active tool message preserved in full fidelity
        self.assertIn(large_tool, trimmed)
        self.assertIn(ai_req, trimmed)


if __name__ == "__main__":
    unittest.main()



