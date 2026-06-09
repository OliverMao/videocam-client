import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

MEMORY_INSTRUCTION = """

**记忆系统：**
你可以使用 memory_write 工具将重要信息保存到长期记忆。当用户说"记住"、"请记住"或表达类似意图时，必须调用 memory_write。
系统会自动在下方注入记忆索引和相关历史记忆供你参考。回答时可参考这些记忆内容，但不要向用户提及记忆系统的存在。"""

CONSOLIDATION_THRESHOLD = 10
MAX_MEMORIES_PER_QUERY = 5
MAX_MEMORY_CHARS = 2000
MAX_IMPLICIT_EXTRACTS = 3


class MemoryManager:
    def __init__(
        self,
        memory_dir: Path,
        client: AsyncOpenAI,
        model: str,
    ):
        self.memory_dir = memory_dir
        self.client = client
        self.model = model
        self._index_cache: str | None = None
        self._index_mtime: float = 0
        self._ready = False

    async def ensure_ready(self) -> None:
        if self._ready:
            return
        try:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            index_path = self.memory_dir / "MEMORY.md"
            if not index_path.exists():
                index_path.write_text("", encoding="utf-8")
            self._ready = True
            logger.info(f"[Memory] 就绪: {self.memory_dir}")
        except Exception as e:
            logger.error(f"[Memory] 初始化失败: {e}")

    async def write_memory(
        self,
        content: str,
        type_: str = "reference",
        tags: list[str] | None = None,
    ) -> str:
        await self.ensure_ready()
        mem_id = str(int(time.time() * 1000))
        meta = {
            "id": mem_id,
            "type": type_,
            "created": datetime.now(timezone.utc).isoformat(),
            "tags": tags or [],
        }
        filepath = self.memory_dir / f"{mem_id}.md"
        filepath.write_text(self._format_frontmatter(meta, content), encoding="utf-8")
        logger.info(f"[Memory] 写入: {mem_id} ({type_}) {content[:60]}...")
        await self.rebuild_index()

        file_count = len(list(self.memory_dir.glob("*.md"))) - 1
        if file_count >= CONSOLIDATION_THRESHOLD:
            asyncio.create_task(self._consolidate_bg())

        return mem_id

    async def read_memory(self, memory_id: str) -> dict[str, Any] | None:
        filepath = self.memory_dir / f"{memory_id}.md"
        if not filepath.exists():
            return None
        try:
            text = filepath.read_text(encoding="utf-8")
            meta, body = self._parse_frontmatter(text)
            return {"id": memory_id, "meta": meta, "content": body.strip()}
        except Exception as e:
            logger.warning(f"[Memory] 读取失败 {memory_id}: {e}")
            return None

    async def delete_memory(self, memory_id: str) -> None:
        filepath = self.memory_dir / f"{memory_id}.md"
        filepath.unlink(missing_ok=True)
        await self.rebuild_index()

    async def rebuild_index(self) -> None:
        files = sorted(
            (f for f in self.memory_dir.glob("*.md") if f.name != "MEMORY.md"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        rows: list[str] = []
        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
                meta, body = self._parse_frontmatter(text)
                mid = meta.get("id", f.stem)
                mtype = meta.get("type", "reference")
                tags = ", ".join(meta.get("tags", []))
                summary = body.strip().replace("\n", " ")[:40]
                rows.append(f"| {mid} | {mtype} | {tags} | {summary} |")
            except Exception:
                continue

        if rows:
            header = "| ID | Type | Tags | Summary |\n|----|------|------|---------|\n"
            index_text = header + "\n".join(rows)
        else:
            index_text = ""

        index_path = self.memory_dir / "MEMORY.md"
        index_path.write_text(index_text, encoding="utf-8")
        self._index_cache = index_text
        self._index_mtime = index_path.stat().st_mtime

    def _get_cached_index(self) -> str:
        index_path = self.memory_dir / "MEMORY.md"
        if not index_path.exists():
            return ""
        try:
            mtime = index_path.stat().st_mtime
            if self._index_cache is not None and mtime == self._index_mtime:
                return self._index_cache
            self._index_cache = index_path.read_text(encoding="utf-8")
            self._index_mtime = mtime
            return self._index_cache
        except Exception:
            return ""

    async def load_relevant(self, query: str) -> list[dict[str, Any]]:
        await self.ensure_ready()
        index_text = self._get_cached_index()
        if not index_text.strip():
            return []

        memory_ids: list[str] = []
        try:
            memory_ids = await self._select_relevant_llm(query, index_text)
        except Exception as e:
            logger.debug(f"[Memory] LLM 选择失败, 回退关键词: {e}")

        if not memory_ids:
            memory_ids = await self._select_relevant_keyword(query)

        memories: list[dict[str, Any]] = []
        for mid in memory_ids[:MAX_MEMORIES_PER_QUERY]:
            mem = await self.read_memory(mid)
            if mem:
                memories.append(mem)

        return self._truncate_memories(memories)

    async def _select_relevant_llm(self, query: str, index_text: str) -> list[str]:
        prompt = (
            "从记忆索引中选出与用户问题最相关的记忆ID（最多5个）。\n"
            "只返回JSON数组如 [\"1704067200000\"]，无相关则返回 []。\n\n"
            f"用户问题：{query}\n\n记忆索引：\n{index_text}\n\n相关ID："
        )
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r"\[.*?]", text, re.DOTALL)
        if not match:
            return []
        ids = json.loads(match.group())
        return ids if isinstance(ids, list) else []

    async def _select_relevant_keyword(self, query: str) -> list[str]:
        files = [f for f in self.memory_dir.glob("*.md") if f.name != "MEMORY.md"]
        query_chars = set(query.lower())
        scored: list[tuple[float, str]] = []
        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
                meta, body = self._parse_frontmatter(text)
                tags = " ".join(meta.get("tags", [])).lower()
                combined = tags + " " + body[:200].lower()
                score = len(query_chars & set(combined)) / max(len(query_chars), 1)
                scored.append((score, meta.get("id", f.stem)))
            except Exception:
                continue
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mid for _, mid in scored[:MAX_MEMORIES_PER_QUERY] if _ > 0]

    def _truncate_memories(self, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        total = 0
        for m in memories:
            size = len(m.get("content", ""))
            if total + size > MAX_MEMORY_CHARS:
                break
            result.append(m)
            total += size
        return result

    def build_system_prompt(
        self, base_prompt: str, relevant_memories: list[dict[str, Any]]
    ) -> str:
        parts = [base_prompt + MEMORY_INSTRUCTION]

        index_text = self._get_cached_index()
        if index_text.strip():
            parts.append(f"\n\n**记忆索引：**\n{index_text}")

        if relevant_memories:
            lines: list[str] = []
            for m in relevant_memories:
                mtype = m.get("meta", {}).get("type", "reference")
                tags = ", ".join(m.get("meta", {}).get("tags", []))
                tag_str = f" (tags: {tags})" if tags else ""
                lines.append(f"- [{mtype}]{tag_str} {m['content']}")
            parts.append("\n\n**相关历史记忆：**\n" + "\n".join(lines))

        return "".join(parts)

    async def extract_from_conversation(
        self, question: str, answer: str
    ) -> None:
        existing = self._get_cached_index()
        prompt = (
            "你是一个信息提取器。从以下对话中提取值得长期记住的信息。\n\n"
            "提取规则：\n"
            "- 用户表达的偏好或要求 → type: user\n"
            "- 用户对回答的纠正或反馈 → type: feedback\n"
            "- 项目或设备相关的事实 → type: project\n"
            "- 其他有参考价值的知识 → type: reference\n"
            "- 如果没有值得记住的内容，返回空数组 []\n\n"
            f"已有记忆：\n{existing[:500]}\n\n"
            f"用户问题：{question}\n助手回答：{answer[:500]}\n\n"
            '只返回JSON数组：[{"content": "...", "type": "user|feedback|project|reference", "tags": ["..."]}]'
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            match = re.search(r"\[.*]", text, re.DOTALL)
            if not match:
                return
            memories = json.loads(match.group())
            if not isinstance(memories, list):
                return
            for mem in memories[:MAX_IMPLICIT_EXTRACTS]:
                if isinstance(mem, dict) and "content" in mem:
                    await self.write_memory(
                        content=mem["content"],
                        type_=mem.get("type", "reference"),
                        tags=mem.get("tags", []),
                    )
                    logger.info(f"[Memory] 隐式提取: {mem['content'][:60]}")
        except Exception as e:
            logger.debug(f"[Memory] 提取跳过: {e}")

    async def consolidate(self) -> None:
        files = [f for f in self.memory_dir.glob("*.md") if f.name != "MEMORY.md"]
        if len(files) < CONSOLIDATION_THRESHOLD:
            return

        logger.info(f"[Memory] 合并: {len(files)} 个文件")

        by_type: dict[str, list[dict[str, Any]]] = {}
        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
                meta, content = self._parse_frontmatter(text)
                type_ = meta.get("type", "reference")
                by_type.setdefault(type_, []).append({
                    "id": meta.get("id", f.stem),
                    "file": f,
                    "content": content.strip(),
                    "tags": meta.get("tags", []),
                })
            except Exception:
                continue

        for type_, memories in by_type.items():
            if len(memories) < 2:
                continue

            mem_text = "\n".join(
                f"- ID:{m['id']}: {m['content']} (tags: {m['tags']})"
                for m in memories
            )
            prompt = (
                f"合并以下同类记忆（类型: {type_}），去除重复，保留重要信息。\n"
                '返回JSON数组：[{"content": "...", "tags": ["..."]}]\n\n'
                f"原始记忆：\n{mem_text}\n\n合并结果："
            )
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0,
                )
                text = response.choices[0].message.content.strip()
                match = re.search(r"\[.*]", text, re.DOTALL)
                if not match:
                    continue
                merged = json.loads(match.group())
                if not isinstance(merged, list) or not merged:
                    continue

                for m in memories:
                    m["file"].unlink(missing_ok=True)
                for item in merged:
                    if isinstance(item, dict) and "content" in item:
                        await self.write_memory(
                            content=item["content"],
                            type_=type_,
                            tags=item.get("tags", []),
                        )
                logger.info(f"[Memory] 合并 {type_}: {len(memories)} → {len(merged)}")
            except Exception as e:
                logger.warning(f"[Memory] 合并 {type_} 失败: {e}")

        await self.rebuild_index()

    async def _consolidate_bg(self) -> None:
        try:
            await self.consolidate()
        except Exception as e:
            logger.warning(f"[Memory] 后台合并失败: {e}")

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict, str]:
        if not text.startswith("---"):
            return {}, text
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}, text
        try:
            meta = yaml.safe_load(parts[1]) or {}
        except yaml.YAMLError:
            meta = {}
        return meta, parts[2]

    @staticmethod
    def _format_frontmatter(meta: dict, content: str) -> str:
        yaml_str = yaml.dump(meta, allow_unicode=True, default_flow_style=False).strip()
        return f"---\n{yaml_str}\n---\n\n{content}\n"
