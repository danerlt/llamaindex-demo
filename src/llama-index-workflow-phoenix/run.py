import asyncio
import os
import subprocess
import sys

from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike
from llama_index.utils.workflow import draw_all_possible_flows
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from workflow import TranslateWorkflow

from phoenix.otel import register


async def main():
    load_dotenv()
    model = os.environ.get("MODEL")
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE")
    llm = OpenAILike(model=model, api_base=api_base, api_key=api_key, is_chat_model=True)

    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
    tracer_provider = register(project_name="default", endpoint=endpoint)
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

    workflow = TranslateWorkflow(
        llm=llm, verbose=True, timeout=240.0
    )
    # draw_all_possible_flows(workflow, filename="translate_workflow.html")
    topic = """
    A Workflow in LlamaIndex is an event-driven abstraction used to chain together several events. Workflows are made up of steps, with each step responsible for handling certain event types and emitting new events.

Workflows in LlamaIndex work by decorating function with a @step decorator. This is used to infer the input and output types of each workflow for validation, and ensures each step only runs when an accepted event is ready.

You can create a Workflow to do anything! Build an agent, a RAG flow, an extraction flow, or anything else you want.

Workflows are also automatically instrumented, so you get observability into each step using tools like Arize Pheonix. (NOTE: Observability works for integrations that take advantage of the newer instrumentation system. Usage may vary.)
    """
    result = await workflow.run(query=topic)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
