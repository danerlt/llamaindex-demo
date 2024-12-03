from typing import Any, List

from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import Document
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)


class BeginTranslateEvent(Event):
    query: str


class LiteralTranslationResultEvent(Event):
    result: str


class FreeTranslationResult(Event):
    result: str


class TranslateWorkflow(Workflow):
    def __init__(
            self,
            *args: Any,
            llm: LLM,
            **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.visited_urls: set[str] = set()

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> BeginTranslateEvent:
        query = ev.query
        await ctx.set("query", query)
        return BeginTranslateEvent(query=query)

    @step
    async def literal_translation(
            self, ctx: Context, ev: BeginTranslateEvent
    ) -> LiteralTranslationResultEvent:
        """ 直译

        """
        query = ev.query
        chat_response = self.llm.chat([
            ChatMessage(role="system", content="请给我直译用户提供的输入。"),
            ChatMessage(role="user", content=f"输入：\n{query}")
        ])
        result = chat_response.message.content
        return LiteralTranslationResultEvent(result=result)

    @step
    async def free_translation(self, ctx: Context, ev: BeginTranslateEvent) -> FreeTranslationResult:
        """ 意译

        """
        query = ev.query
        chat_response = self.llm.chat([
            ChatMessage(role="system", content="请给我意译用户提供的输入。"),
            ChatMessage(role="user", content=f"输入：\n{query}")
        ])
        result = chat_response.message.content
        return FreeTranslationResult(result=result)

    @step
    async def merge(self, ctx: Context, ev: LiteralTranslationResultEvent | FreeTranslationResult) -> StopEvent:
        data = ctx.collect_events(ev, [LiteralTranslationResultEvent, FreeTranslationResult])
        # check if we can run
        if data is None:
            return StopEvent()

        # unpack -- data is returned in order
        literal_translation_result_event, free_translation_result = data
        #  直译结果
        query = await ctx.get("query")

        chat_response = self.llm.chat([
            ChatMessage(role="system", content="请给我综合原文和直译和意译的结果进行翻译"),
            ChatMessage(role="user", content=f"原文：\n{query}\n\n直译结果:\n{literal_translation_result_event.result}\n\n意译结果:\n{free_translation_result.result}")
        ])
        result = chat_response.message.content
        return StopEvent(result=result)
