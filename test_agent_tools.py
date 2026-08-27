import os
import unittest
from pathlib import Path

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
    _RECORDED_FINDINGS,
    MODULE_AUDIT_TOOLS
)

# Explainer Agent
from code_explainer_agent import build_explainer_graph

from langchain_core.messages import AIMessage


class TestCodeReviewAndExplainerTools(unittest.TestCase):
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
        result = graph.invoke({"messages": [{"role": "user", "content": "Explain OrderRepository"}]})
        self.assertIn("messages", result)
        self.assertEqual(result["messages"][-1].content, "Here is how the architecture works.")


if __name__ == "__main__":
    unittest.main()
