import json

from src.agent_tools import ReportsRAGTool

def test_reports_rag_tool_returns_matches():
    tool = ReportsRAGTool()
    result = tool(query="Myelom", report_type="Arztbrief")
    data = json.loads(result.content)
    nodes = data.get("context_nodes")
    assert isinstance(nodes, list) and len(nodes) > 0
    assert any(node.get("patient_id") == tool.patient_id for node in nodes)
