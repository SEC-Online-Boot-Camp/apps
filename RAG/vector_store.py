import json
import os

import numpy as np


class VectorStore:
    """チャンク(文書の断片)とそのベクトルを保持し、類似検索を行う"""

    def __init__(self):
        self._chunks: list[str] = []
        self._sources: list[str] = []
        self._vectors: np.ndarray | None = None

    def add(self, chunks: list[str], sources: list[str], vectors: list[list[float]]):
        """チャンクとそのベクトルを保存する"""
        self._chunks.extend(chunks)
        self._sources.extend(sources)
        new_vectors = np.array(vectors, dtype=np.float32)
        if self._vectors is None:
            self._vectors = new_vectors
        else:
            self._vectors = np.vstack([self._vectors, new_vectors])

    def search(self, query_vector: list[float], top_k: int = 3) -> list[dict]:
        """質問のベクトルに近いチャンクを返す"""
        if self._vectors is None:
            return []
        q = np.array(query_vector, dtype=np.float32)

        doc_norms = self._vectors / np.linalg.norm(self._vectors, axis=1, keepdims=True)
        q_norm = q / np.linalg.norm(q)
        similarities = doc_norms @ q_norm
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "chunk": self._chunks[idx],
                    "source": self._sources[idx],
                    "score": float(similarities[idx]),
                }
            )
        return results

    def save(self, cache_dir: str):
        """キャッシュを保存する"""
        os.makedirs(cache_dir, exist_ok=True)
        np.save(os.path.join(cache_dir, "vectors.npy"), self._vectors)
        with open(os.path.join(cache_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"chunks": self._chunks, "sources": self._sources},
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, cache_dir: str) -> "VectorStore":
        """ファイルからベクトルストアを復元する"""
        store = cls()
        store._vectors = np.load(os.path.join(cache_dir, "vectors.npy"))
        with open(os.path.join(cache_dir, "metadata.json"), encoding="utf-8") as f:
            meta = json.load(f)
        store._chunks = meta["chunks"]
        store._sources = meta["sources"]
        return store

    @staticmethod
    def exists(cache_dir: str) -> bool:
        return os.path.isfile(
            os.path.join(cache_dir, "vectors.npy")
        ) and os.path.isfile(os.path.join(cache_dir, "metadata.json"))

    def items(self) -> list[dict]:
        """保存されているチャンクを {"index", "source", "text"} のリストで返す。"""
        return [
            {"index": i, "source": self._sources[i], "text": self._chunks[i]}
            for i in range(len(self._chunks))
        ]

    def __len__(self) -> int:
        return len(self._chunks)
