import dspy
from llama_index.core.llms import ChatMessage
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms.function_calling import FunctionCallingLLM
from typing import Optional

dspy_lm = dspy.LM( 
    "openai/casperhansen/llama-3.3-70b-instruct-awq", 
    api_base="http://localhost:9999/v1", 
    api_key="dacbebe8c973154018a3d0f5" 
    ) 
dspy.configure(lm=dspy_lm) 
qa = dspy.ChainOfThought('question -> answer') 
def answer_question(question: str) -> str: 
    return qa(question=question).answer

def calculate(a: float, b: float, operator: str) -> str:
    """Simple calculus on two numbers a and b. Supports +, -, *, /."""
    match operator:
        case "+":
            return f"The sum of {a} and {b} is {a+b}."
        case "-":
            return f"Subtracting {a} and {b} (a-b) is {a-b}."
        case "*":
            return f"Multiplying {a} and {b} is {a*b}."
        case "/":
            return f"The ratio between {a} and {b} is {a/b}."
        case _:
            return "Invalid operator. Please use one of the following: +, -, *, /."

async def main():
    llm = OpenAILike(model="casperhansen/llama-3.3-70b-instruct-awq", api_base="http://localhost:9999/v1", api_key="dacbebe8c973154018a3d0f5")
    tools = [
        FunctionTool.from_defaults(fn=answer_question),
        FunctionTool.from_defaults(fn=calculate),
    ]
    workflow = AgentWorkflow.from_tools_or_functions(
        tools, llm=llm, system_prompt="You are a helpful assistant."
    )

    resp = await workflow.run(user_msg="Calculate 10 + 5")
    print("Workflow response:", resp)
    for tool_call in resp.__dict__.get("tool_calls", []):
        print(f"Tool used: {tool_call.tool_name}, args: {tool_call.tool_kwargs}, output: {tool_call.tool_output.raw_output}")

    resp2 = await workflow.run(user_msg="What is the capital of France?")
    print("Workflow response:", resp2)
    for tool_call in resp2.__dict__.get("tool_calls", []):
        print(f"Tool used: {tool_call.tool_name}, args: {tool_call.tool_kwargs}, output: {tool_call.tool_output.raw_output}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())