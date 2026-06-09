import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from agent.memory import MemoryManager
from agent.prompt import SYSTEM_PROMPT
from agent.tools import MAX_TOOL_ITERATIONS, TOOL_DESCRIPTIONS, TOOLS

logger = logging.getLogger(__name__)


class QAAgent:
    def __init__(
        self,
        base_url: str = "http://116.238.240.2:30630",
        api_key: str = "vllm",
        model: str = "Qwen3.6-35B-A3B",
        vectordb=None,
    ):
        logger.info(f"[QA] QAAgent 初始化: model={model}, base_url={base_url}")
        self.client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key=api_key)
        self.model = model
        self.vectordb = vectordb
        self.memory = MemoryManager(
            memory_dir=Path(__file__).parent.parent / ".memory",
            client=self.client,
            model=model,
        )

    async def execute_tool(self, name: str, input_data: dict) -> str:
        t0 = time.monotonic()
        logger.info(f"[QA][Tool] 开始执行: {name}, 输入: {input_data}")
        result = None
        if name == "todo_write":
            todos = input_data.get("todos", [])
            result = json.dumps({"status": "ok", "message": f"已更新 {len(todos)} 个任务"}, ensure_ascii=False)
        elif name == "query_database":
            start_time = input_data.get("start_time", "")
            end_time = input_data.get("end_time", "")
            keywords = input_data.get("keywords", [])
            frames: list[dict] = []
            if self.vectordb and start_time and end_time:
                try:
                    frames = self.vectordb.search_by_time(start_time, end_time, limit=200)
                except Exception as e:
                    logger.warning(f"[QA][Tool] VectorDB 查询失败: {e}")
            if frames:
                frame_summaries = [
                    {"id": f["id"], "timestamp": f["timestamp"], "thumb": f"/api/vectordb/thumb/{f.get('thumb', '')}"}
                    for f in frames
                ]
                result = json.dumps({
                    "status": "ok",
                    "message": f"查询区间 {start_time} ~ {end_time}，关键词：{keywords}，共找到 {len(frames)} 帧",
                    "data": frame_summaries,
                    "total": len(frames),
                }, ensure_ascii=False)
            else:
                result = json.dumps({
                    "status": "ok",
                    "message": f"查询区间 {start_time} ~ {end_time}，关键词：{keywords}，矢量数据库中暂无匹配帧",
                    "data": [],
                }, ensure_ascii=False)
        elif name == "sort_by_relevance":
            raw_data = input_data.get("data", "")
            query_desc = input_data.get("query", "")
            try:
                parsed = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
                items = parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, TypeError):
                items = []
            if items:
                sorted_items = sorted(items, key=lambda x: x.get("timestamp", ""), reverse=True)
                result = json.dumps({
                    "status": "ok",
                    "message": f"按「{query_desc}」排序完成，共 {len(sorted_items)} 条",
                    "data": sorted_items,
                }, ensure_ascii=False)
            else:
                result = json.dumps({
                    "status": "ok",
                    "message": f"按「{query_desc}」排序完成，无数据",
                }, ensure_ascii=False)
        elif name == "summarize":
            result = json.dumps({
                "status": "ok",
                "message": f"已按「{input_data.get('style','简洁')}」风格总结",
            }, ensure_ascii=False)
        elif name == "memory_write":
            try:
                mem_id = await self.memory.write_memory(
                    content=input_data.get("content", ""),
                    type_=input_data.get("type", "reference"),
                    tags=input_data.get("tags", []),
                )
                result = json.dumps({"status": "ok", "message": f"已保存记忆 (id={mem_id})"}, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"[QA][Tool] memory_write 失败: {e}")
                result = json.dumps({"status": "error", "message": f"保存失败: {e}"}, ensure_ascii=False)
        else:
            logger.warning(f"[QA][Tool] 未知工具: {name}")
            result = json.dumps({"status": "error", "message": f"未知工具: {name}"})

        elapsed = (time.monotonic() - t0) * 1000
        logger.info(f"[QA][Tool] {name} 完成, 耗时 {elapsed:.1f}ms, 结果: {result[:120]}...")
        return result

    async def _call_ai(self, messages: list, is_first_call: bool, system_prompt: str | None = None):
        phase = "Phase1" if is_first_call else "Phase2"
        t0 = time.monotonic()
        logger.info(f"[QA][{phase}] 调用 AI: messages={len(messages)}, model={self.model}")

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt or SYSTEM_PROMPT}] + messages,
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

        try:
            await self.memory.ensure_ready()
            relevant = await self.memory.load_relevant(question)
            system_prompt = self.memory.build_system_prompt(SYSTEM_PROMPT, relevant)
            if relevant:
                logger.info(f"[QA] 注入 {len(relevant)} 条相关记忆")
        except Exception as e:
            logger.warning(f"[QA] 记忆加载失败: {e}")
            system_prompt = None

        result: dict[str, Any] = {"answer": "", "steps": []}
        todo_step_idx: int | None = None

        def add_step(type_: str, title: str, content: str) -> None:
            result["steps"].append({"type": type_, "title": title, "content": content})

        try:
            logger.info("[QA] Phase 1: 分析问题...")
            text, tool_calls, assistant_message = await self._call_ai(messages, is_first_call=True, system_prompt=system_prompt)

            if text:
                add_step("thinking", "分析问题", text)

            logger.info(f"[QA] Phase 1 完成: text_len={len(text)}, tool_calls={len(tool_calls)}, elapsed={((time.monotonic()-t0)*1000):.1f}ms")

            if not tool_calls:
                logger.info("[QA] 无工具调用, 直接输出回答")
                add_step("answer", "生成回答", text)
                result["answer"] = text
                asyncio.create_task(self._extract_memories_bg(question, text))
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

                text, tool_calls, assistant_message = await self._call_ai(messages, is_first_call=False, system_prompt=system_prompt)
                logger.info(f"[QA] 迭代 {tool_iter+1} 完成: text_len={len(text)}, new_tool_calls={len(tool_calls)}, elapsed={((time.monotonic()-t0)*1000):.1f}ms")
                messages.append(assistant_message)

                if not tool_calls:
                    logger.info("[QA] 无更多工具调用, 输出最终回答")
                    add_step("answer", "生成回答", text)
                    result["answer"] = text
                    asyncio.create_task(self._extract_memories_bg(question, text))
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

        try:
            await self.memory.ensure_ready()
            relevant = await self.memory.load_relevant(question)
            system_prompt = self.memory.build_system_prompt(SYSTEM_PROMPT, relevant)
            if relevant:
                logger.info(f"[QA][Stream] 注入 {len(relevant)} 条相关记忆")
        except Exception as e:
            logger.warning(f"[QA][Stream] 记忆加载失败: {e}")
            system_prompt = None

        todo_step_idx: int | None = None
        step_count = 0

        def new_step(type_: str, title: str, content: str) -> dict:
            nonlocal step_count, todo_step_idx
            idx = step_count
            step_count += 1
            return {"idx": idx, "type": type_, "title": title, "content": content}

        try:
            analysis_idx = step_count
            step_count += 1
            yield self._sse({"event": "step_add", "step": {"idx": analysis_idx, "type": "analysis", "title": "分析问题", "content": "", "status": "active"}})

            text, tool_calls, assistant_message = await self._call_ai(messages, is_first_call=True, system_prompt=system_prompt)

            logger.info(f"[QA][Stream] Phase1 done: text_len={len(text)}, tools={len(tool_calls)}")

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
                yield self._sse({"event": "step_add", "step": {"idx": step_count, "type": "answer", "title": "生成回答", "content": "", "status": "active"}})
                answer_idx = step_count
                step_count += 1
                async for chunk in self._stream_text(text):
                    yield self._sse({"event": "answer_chunk", "idx": answer_idx, "text": chunk})
                yield self._sse({"event": "step_status", "idx": answer_idx, "status": "done"})
                yield self._sse({"event": "done"})
                asyncio.create_task(self._extract_memories_bg(question, text))
                return

            messages.append(assistant_message)

            active_tool_calls = remaining_tool_calls
            if pending_todos and not active_tool_calls:
                text, tool_calls, assistant_message = await self._call_ai(messages, is_first_call=False, system_prompt=system_prompt)
                messages.append(assistant_message)
                active_tool_calls = tool_calls

            for tool_iter in range(MAX_TOOL_ITERATIONS):
                logger.info(f"[QA][Stream] --- 工具迭代 {tool_iter+1} ---")

                for tc in active_tool_calls:
                    if tc["name"] == "todo_write":
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

                text, tool_calls, assistant_message = await self._call_ai(messages, is_first_call=False, system_prompt=system_prompt)
                messages.append(assistant_message)
                active_tool_calls = tool_calls

                if not active_tool_calls:
                    yield self._sse({"event": "step_add", "step": {"idx": step_count, "type": "answer", "title": "生成回答", "content": "", "status": "active"}})
                    answer_idx = step_count
                    step_count += 1
                    async for chunk in self._stream_text(text):
                        yield self._sse({"event": "answer_chunk", "idx": answer_idx, "text": chunk})
                    yield self._sse({"event": "step_status", "idx": answer_idx, "status": "done"})
                    yield self._sse({"event": "done"})
                    asyncio.create_task(self._extract_memories_bg(question, text))
                    return

                if text:
                    yield self._sse({"event": "step_add", "step": {"idx": step_count, "type": "thinking", "title": "分析中间结果", "content": text, "status": "done"}})
                    step_count += 1

            yield self._sse({"event": "done"})

        except Exception as e:
            logger.error(f"[QA][Stream] 异常: {e}", exc_info=True)
            yield self._sse({"event": "error", "message": str(e)})

    async def _extract_memories_bg(self, question: str, answer: str) -> None:
        try:
            await self.memory.extract_from_conversation(question, answer)
        except Exception as e:
            logger.debug(f"[QA] 记忆提取失败: {e}")

    async def _stream_text(self, text: str):
        """Fake-stream existing text 4 chars at a time."""
        chunk_size = 4
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
            await asyncio.sleep(0.02)

    async def _stream_answer(self, messages: list, system_prompt: str | None = None):
        """Call AI with stream=True and yield content chunks."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt or SYSTEM_PROMPT}] + messages,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as e:
            logger.error(f"[QA][Stream] _stream_answer 异常: {e}", exc_info=True)
            yield f"[错误: {e}]"
