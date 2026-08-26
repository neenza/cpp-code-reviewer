import os
import unittest
from pathlib import Path
from code_review_agent import (
    clangd_query,
    ripgrep_search,
    read_project_file,
    set_active_project_dir,
    build_code_review_graph,
    TOOLS
)
from langchain_core.messages import AIMessage

class TestCodeReviewTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_dir = Path(__file__).parent / "sample_project"
        set_active_project_dir(cls.sample_dir)

    def test_read_cmakelists(self):
        content = read_project_file.invoke({"file_path": "CMakeLists.txt"})
        self.assertIn("SampleOrderSystem", content)
        self.assertIn("CMAKE_CXX_STANDARD 17", content)

    def test_ripgrep_search(self):
        res = ripgrep_search.invoke({"pattern": "strcpy"})
        self.assertIn("session_manager.cpp", res)

        res_mutex = ripgrep_search.invoke({"pattern": "std::shared_mutex"})
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
        from code_review_agent import record_finding, list_project_source_files, _RECORDED_FINDINGS
        res = record_finding.invoke({
            "category": "critical_flaw",
            "title": "Buffer overflow in SessionManager",
            "details": "strcpy without bounds check",
            "files_and_lines": "src/session_manager.cpp:18",
            "recommended_fix": "Use std::string or snprintf"
        })
        self.assertIn("CRITICAL_FLAW", res)
        self.assertTrue(any(f["title"] == "Buffer overflow in SessionManager" for f in _RECORDED_FINDINGS))

        file_list = list_project_source_files.invoke({})
        self.assertIn("order_repository.cpp", file_list)
        self.assertIn("session_manager.h", file_list)

        from code_review_agent import track_review_progress
        prog = track_review_progress.invoke({"inspected_files": ["src/order_repository.cpp"]})
        self.assertIn("Code Review Coverage", prog)

    def test_langgraph_compilation(self):
        class DummyLLM:
            def bind_tools(self, tools):
                return self
            def invoke(self, messages):
                return AIMessage(content="Review complete.")

        graph = build_code_review_graph(llm=DummyLLM(), tools=TOOLS)
        self.assertIsNotNone(graph)
        result = graph.invoke({"messages": [{"role": "user", "content": "Review project"}]})
        self.assertIn("messages", result)
        self.assertEqual(result["messages"][-1].content, "Review complete.")

if __name__ == "__main__":
    unittest.main()
