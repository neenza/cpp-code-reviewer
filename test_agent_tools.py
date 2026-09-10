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
        self.assertIn("## Order Repository Architecture", content)
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


if __name__ == "__main__":
    unittest.main()
