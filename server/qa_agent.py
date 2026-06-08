import asyncio
import json
import logging
import time
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个智能监控数据分析助手，负责基于历史监控数据回答用户问题。

你的工作流程（必须严格遵守）：
1. **分析问题** — 理解用户的核心意图，输出一段分析文字
2. **制定计划** — 立即调用 todo_write，列出所有待执行任务（全部 pending）
3. **逐步执行** — 按顺序执行每个任务：执行前调用 todo_write 将该任务改为 in_progress，执行后改为 completed，然后继续下一个

可用工具：
- todo_write：更新任务列表（每次执行前后都要更新状态）
- query_database：按时间区间和关键词查询历史监控数据
- sort_by_relevance：按匹配度对查询结果排序
- summarize：对最终结果进行总结表达

要求：第一步必须是 todo_write 列出计划，未经计划不得直接调用其他工具。"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": "创建或更新任务列表，在动手前列出所有步骤，执行时逐步更新状态",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "完整的任务列表（每次调用必须包含全部任务）",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "任务描述"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"], "description": "任务状态"},
                            },
                            "required": ["content", "status"],
                        },
                    },
                },
                "required": ["todos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "按时间区间和关键词查询历史监控数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "开始时间，格式 YYYY-MM-DD HH:mm:ss"},
                    "end_time": {"type": "string", "description": "结束时间，格式 YYYY-MM-DD HH:mm:ss"},
                    "keywords": {"type": "array", "items": {"type": "string"}, "description": "搜索关键词，如吸烟、打架等"},
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sort_by_relevance",
            "description": "按匹配度对查询结果进行排序",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "待排序的原始数据"},
                    "query": {"type": "string", "description": "排序依据的查询描述"},
                },
                "required": ["data", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "对最终结果进行总结表达",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "待总结的内容"},
                    "style": {"type": "string", "description": "表达风格，可选：简洁、详细、报告"},
                },
                "required": ["content"],
            },
        },
    },
]

TOOL_DESCRIPTIONS: dict[str, str] = {
    t["function"]["name"]: t["function"]["description"]
    for t in TOOLS
}

MAX_TOOL_ITERATIONS = 10


class QAAgent:
    def __init__(
        self,
        base_url: str = "http://116.238.240.2:30630",
        api_key: str = "vllm",
        model: str = "Qwen3.6-35B-A3B",
    ):
        logger.info(f"[QA] QAAgent 初始化: model={model}, base_url={base_url}")
        self.client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key=api_key)
        self.model = model

    async def execute_tool(self, name: str, input_data: dict) -> str:
        t0 = time.monotonic()
        logger.info(f"[QA][Tool] 开始执行: {name}, 输入: {input_data}")
        result = None
        if name == "todo_write":
            todos = input_data.get("todos", [])
            result = json.dumps({"status": "ok", "message": f"已更新 {len(todos)} 个任务"}, ensure_ascii=False)
        elif name == "query_database":
            result = json.dumps({
                "status": "ok",
                "message": f"查询区间 {input_data.get('start_time','')} ~ {input_data.get('end_time','')}，关键词：{input_data.get('keywords',[])}",
                "data": [],
            }, ensure_ascii=False)
        elif name == "sort_by_relevance":
            result = json.dumps({
                "status": "ok",
                "message": f"按「{input_data.get('query','')}」排序完成",
            }, ensure_ascii=False)
        elif name == "summarize":
            result = json.dumps({
                "status": "ok",
                "message": f"已按「{input_data.get('style','简洁')}」风格总结",
            }, ensure_ascii=False)
        else:
            logger.warning(f"[QA][Tool] 未知工具: {name}")
            result = json.dumps({"status": "error", "message": f"未知工具: {name}"})

        elapsed = (time.monotonic() - t0) * 1000
        logger.info(f"[QA][Tool] {name} 完成, 耗时 {elapsed:.1f}ms, 结果: {result[:120]}...")
        return result

    async def _call_ai(self, messages: list, is_first_call: bool):
        phase = "Phase1" if is_first_call else "Phase2"
        t0 = time.monotonic()
        logger.info(f"[QA][{phase}] 调用 AI: messages={len(messages)}, model={self.model}")

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            tools=TOOLS,
        )

        choice = response.choices[0]
        message = choice.message
        text = message.content or ""

        parsed_tools = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    input_data = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    logger.warning(f"[QA][{phase}] 工具 {tc.function.name} 参数解析失败: {tc.function.arguments[:100]}")
                    input_data = {}
                parsed_tools.append({"id": tc.id, "name": tc.function.name, "input": input_data})
                logger.info(f"[QA][{phase}] 发现工具调用: {tc.function.name}(id={tc.id}), input={json.dumps(input_data, ensure_ascii=False)[:120]}")

        if text:
            logger.info(f"[QA][{phase}] AI 文本输出: {text[:200]}...")

        assistant_message: dict[str, Any] = {"role": "assistant", "content": text or None}
        if parsed_tools:
            assistant_message["tool_calls"] = [
                {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["input"], ensure_ascii=False)}}
                for tc in parsed_tools
            ]

        total_ms = (time.monotonic() - t0) * 1000
        logger.info(f"[QA][{phase}] _call_ai 完成: text_len={len(text)}, tools={len(parsed_tools)}, 总耗时 {total_ms:.1f}ms")
        return text, parsed_tools, assistant_message

    async def ask(self, question: str, history: list | None = None) -> dict[str, Any]:
        t0 = time.monotonic()
        logger.info(f"[QA] ====== 开始新请求: question='{question[:80]}', history_len={len(history or [])} ======")
        messages = list(history or [])
        messages.append({"role": "user", "content": question})

        result: dict[str, Any] = {"answer": "", "steps": []}
        todo_step_idx: int | None = None

        def add_step(type_: str, title: str, content: str) -> None:
            result["steps"].append({"type": type_, "title": title, "content": content})

        try:
            logger.info("[QA] Phase 1: 分析问题...")
            text, tool_calls, assistant_message = await self._call_ai(messages, is_first_call=True)

            if text:
                add_step("thinking", "分析问题", text)

            logger.info(f"[QA] Phase 1 完成: text_len={len(text)}, tool_calls={len(tool_calls)}, elapsed={((time.monotonic()-t0)*1000):.1f}ms")

            if not tool_calls:
                logger.info("[QA] 无工具调用, 直接输出回答")
                add_step("answer", "生成回答", text)
                result["answer"] = text
                return result

            messages.append(assistant_message)

            for tool_iter in range(MAX_TOOL_ITERATIONS):
                logger.info(f"[QA] --- 工具迭代 {tool_iter+1}/{MAX_TOOL_ITERATIONS} ---")

                for tc in tool_calls:
                    logger.info(f"[QA] 执行工具: {tc['name']}(id={tc['id']})")
                    tool_result = await self.execute_tool(tc["name"], tc["input"])

                    if tc["name"] == "todo_write":
                        todos = tc["input"].get("todos", [])
                        todos_json = json.dumps(todos, ensure_ascii=False)
                        if todo_step_idx is None:
                            add_step("todo", "任务规划", todos_json)
                            todo_step_idx = len(result["steps"]) - 1
                        else:
                            result["steps"][todo_step_idx]["content"] = todos_json
                    else:
                        desc = TOOL_DESCRIPTIONS.get(tc["name"], tc["name"])
                        add_step("tool_call", f"智能体正在{desc}", tool_result)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    })

                text, tool_calls, assistant_message = await self._call_ai(messages, is_first_call=False)
                logger.info(f"[QA] 迭代 {tool_iter+1} 完成: text_len={len(text)}, new_tool_calls={len(tool_calls)}, elapsed={((time.monotonic()-t0)*1000):.1f}ms")
                messages.append(assistant_message)

                if not tool_calls:
                    logger.info("[QA] 无更多工具调用, 输出最终回答")
                    add_step("answer", "生成回答", text)
                    result["answer"] = text
                    return result

                if text:
                    add_step("thinking", "分析中间结果", text)

            logger.warning(f"[QA] 超过最大工具迭代次数 ({MAX_TOOL_ITERATIONS})")
            result["answer"] = "超过最大工具迭代次数"
            return result

        except Exception as e:
            logger.error(f"[QA] 异常: {e}", exc_info=True)
            result["answer"] = f"处理出错: {e}"
            return result

    @staticmethod
    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def ask_stream(self, question: str, history: list | None = None):
        t0 = time.monotonic()
        logger.info(f"[QA][Stream] ====== 开始: question='{question[:80]}' ======")
        messages = list(history or [])
        messages.append({"role": "user", "content": question})

        todo_step_idx: int | None = None
        step_count = 0

        def new_step(type_: str, title: str, content: str) -> dict:
            nonlocal step_count, todo_step_idx
            idx = step_count
            step_count += 1
            return {"idx": idx, "type": type_, "title": title, "content": content}

        try:
            # Emit analysis step as active before calling AI
            analysis_idx = step_count
            step_count += 1
            yield self._sse({"event": "step_add", "step": {"idx": analysis_idx, "type": "analysis", "title": "分析问题", "content": "", "status": "active"}})

            text, tool_calls, assistant_message = await self._call_ai(messages, is_first_call=True)

            logger.info(f"[QA][Stream] Phase1 done: text_len={len(text)}, tools={len(tool_calls)}")

            # Collect todos from the first todo_write call (if any) to merge into analysis step
            pending_todos: list = []
            remaining_tool_calls: list = []
            for tc in tool_calls:
                if tc["name"] == "todo_write" and not pending_todos:
                    pending_todos = tc["input"].get("todos", [])
                    await self.execute_tool(tc["name"], tc["input"])
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps({"status": "ok"}, ensure_ascii=False)})
                    todo_step_idx = analysis_idx
                else:
                    remaining_tool_calls.append(tc)

            analysis_content = json.dumps({"thinking": text, "todos": pending_todos}, ensure_ascii=False)
            yield self._sse({"event": "step_update", "idx": analysis_idx, "content": analysis_content, "status": "done"})

            if not tool_calls:
                # No tools — stream answer directly
                yield self._sse({"event": "step_add", "step": {"idx": step_count, "type": "answer", "title": "生成回答", "content": "", "status": "active"}})
                answer_idx = step_count
                step_count += 1
                async for chunk in self._stream_answer(messages + [assistant_message]):
                    yield self._sse({"event": "answer_chunk", "idx": answer_idx, "text": chunk})
                yield self._sse({"event": "step_status", "idx": answer_idx, "status": "done"})
                yield self._sse({"event": "done"})
                return

            messages.append(assistant_message)

            # Process any non-todo tool calls from phase 1
            active_tool_calls = remaining_tool_calls
            if pending_todos and not active_tool_calls:
                # todo_write was the only call; re-ask AI for next steps
                text, tool_calls, assistant_message = await self._call_ai(messages, is_first_call=False)
                messages.append(assistant_message)
                active_tool_calls = tool_calls

            for tool_iter in range(MAX_TOOL_ITERATIONS):
                logger.info(f"[QA][Stream] --- 工具迭代 {tool_iter+1} ---")

                for tc in active_tool_calls:
                    if tc["name"] == "todo_write":
                        # Subsequent todo_write: update analysis step todos
                        todos = tc["input"].get("todos", [])
                        await self.execute_tool(tc["name"], tc["input"])
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps({"status": "ok"}, ensure_ascii=False)})
                        current = json.loads(analysis_content) if analysis_content else {"thinking": text, "todos": []}
                        current["todos"] = todos
                        analysis_content = json.dumps(current, ensure_ascii=False)
                        yield self._sse({"event": "step_update", "idx": analysis_idx, "content": analysis_content, "status": "done"})
                        continue

                    desc = TOOL_DESCRIPTIONS.get(tc["name"], tc["name"])
                    tool_step_idx = step_count
                    step_count += 1
                    yield self._sse({"event": "step_add", "step": {"idx": tool_step_idx, "type": "tool_call", "title": f"智能体正在{desc}", "content": "", "status": "active"}})

                    tool_result = await self.execute_tool(tc["name"], tc["input"])
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_result})

                    yield self._sse({"event": "step_update", "idx": tool_step_idx, "content": tool_result, "status": "done"})

                text, tool_calls, assistant_message = await self._call_ai(messages, is_first_call=False)
                messages.append(assistant_message)
                active_tool_calls = tool_calls

                if not active_tool_calls:
                    # Final answer — stream it
                    yield self._sse({"event": "step_add", "step": {"idx": step_count, "type": "answer", "title": "生成回答", "content": "", "status": "active"}})
                    answer_idx = step_count
                    step_count += 1
                    async for chunk in self._stream_text(text):
                        yield self._sse({"event": "answer_chunk", "idx": answer_idx, "text": chunk})
                    yield self._sse({"event": "step_status", "idx": answer_idx, "status": "done"})
                    yield self._sse({"event": "done"})
                    return

                if text:
                    yield self._sse({"event": "step_add", "step": {"idx": step_count, "type": "thinking", "title": "分析中间结果", "content": text, "status": "done"}})
                    step_count += 1

            yield self._sse({"event": "done"})

        except Exception as e:
            logger.error(f"[QA][Stream] 异常: {e}", exc_info=True)
            yield self._sse({"event": "error", "message": str(e)})

    async def _stream_text(self, text: str):
        """Fake-stream existing text 4 chars at a time."""
        chunk_size = 4
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
            await asyncio.sleep(0.02)

    async def _stream_answer(self, messages: list):
        """Call AI with stream=True and yield content chunks."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as e:
            logger.error(f"[QA][Stream] _stream_answer 异常: {e}", exc_info=True)
            yield f"[错误: {e}]"
