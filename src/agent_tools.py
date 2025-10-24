from llama_index.tools.function_tool import FunctionTool
import requests
import json

RAG_SERVER_URL = "http://localhost:8000"
registered_functions = []
__all__ = ["agent_tools", "tool_names"]
tool_names = {
    "query_patient_record": "Use this tool to search for general information across the patient's entire medical record."
}

def register(func):
    """Decorator func to register a function as an AI agent tool."""
    registered_functions.append(func)
    return func

@register
def query_patient_record(query: str) -> str:
    """
    Searches the entire patient record for information relevant to a given query.
    """
    response = requests.get(f"{RAG_SERVER_URL}/search", params={"query": query, "top_k": 5})
    if response.status_code == 200:
        return json.dumps(response.json())
    else:
        return f"Error: Failed to query patient record with status {response.status_code}"



agent_tools = []
for func in registered_functions:
    tool = FunctionTool.from_defaults(fn=func)
    agent_tools.append(tool)
