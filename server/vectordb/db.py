import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VectorDB:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self._vectors: np.ndarray | None = None
        self._meta: list[dict[str, Any]] = []
        self._dirty = False
        self._load()

    def _load(self) -> None:
        vec_path = self.storage_dir / "vectors.npy"
        meta_path = self.storage_dir / "meta.json"
        if vec_path.exists() and meta_path.exists():
            try:
                self._vectors = np.load(str(vec_path))
                self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
                logger.info(f"[VectorDB] 加载: {len(self._meta)} 条记录, dim={self._vectors.shape[1]}")
            except Exception as e:
                logger.warning(f"[VectorDB] 加载失败: {e}")
                self._vectors = None
                self._meta = []

    def save(self) -> None:
        if not self._dirty:
            return
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        if self._vectors is not None:
            np.save(str(self.storage_dir / "vectors.npy"), self._vectors)
        meta_path = self.storage_dir / "meta.json"
        meta_path.write_text(json.dumps(self._meta, ensure_ascii=False), encoding="utf-8")
        self._dirty = False

    def add(self, vector: np.ndarray, timestamp: str, thumb_path: str = "") -> str:
        vec = vector.reshape(1, -1).astype(np.float32)
        if self._vectors is None:
            self._vectors = vec
        else:
            self._vectors = np.vstack([self._vectors, vec])
        entry_id = str(len(self._meta))
        self._meta.append({
            "id": entry_id,
            "timestamp": timestamp,
            "thumb": thumb_path,
        })
        self._dirty = True
        return entry_id

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        if self._vectors is None or len(self._meta) == 0:
            return []
        q = query_vector.reshape(1, -1).astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        dots = (self._vectors @ q.T).flatten()
        k = min(top_k, len(self._meta))
        top_indices = np.argsort(dots)[::-1][:k]

        results = []
        for idx in top_indices:
            entry = dict(self._meta[idx])
            entry["score"] = float(dots[idx])
            results.append(entry)
        return results

    def search_by_time(self, start: str, end: str, limit: int = 100) -> list[dict[str, Any]]:
        results = []
        for entry in self._meta:
            ts = entry["timestamp"]
            if start <= ts <= end:
                results.append(dict(entry))
                if len(results) >= limit:
                    break
        return results

    def stats(self) -> dict[str, Any]:
        if not self._meta:
            return {"count": 0, "dim": 0, "time_range": None}
        return {
            "count": len(self._meta),
            "dim": self._vectors.shape[1] if self._vectors is not None else 0,
            "time_range": {
                "earliest": self._meta[0]["timestamp"],
                "latest": self._meta[-1]["timestamp"],
            },
        }

    def get_frame(self, entry_id: str) -> dict[str, Any] | None:
        idx = int(entry_id)
        if 0 <= idx < len(self._meta):
            return dict(self._meta[idx])
        return None
