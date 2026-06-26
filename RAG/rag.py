import glob
import os

from embedder import create_embedder
from llm_client import create_llm
from vector_store import VectorStore

_CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "200"))
_OVERLAP = int(os.environ.get("RAG_OVERLAP", "40"))


def _read_file(path: str) -> str:
    """ファイルをテキストとして読み込む"""
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_into_chunks(
    text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _OVERLAP
) -> list[str]:
    """テキストを文字の断片に分割する"""
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)  # overlap >= chunk_size の無限ループ対策
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


class RAG:
    """RAGクラス"""

    def __init__(self):
        self.embedder = create_embedder()
        self.llm = create_llm()
        self.store = VectorStore()

    def reset(self):
        """読み込んだ文書をすべて破棄する"""
        self.store = VectorStore()

    def load_documents(self, folder: str, cache_dir: str | None = None):
        """フォルダ内の .txt / .pdf を読み込む"""
        self.store = VectorStore()

        if cache_dir and VectorStore.exists(cache_dir):
            print(f"[キャッシュ] {cache_dir} から読み込みます...")
            self.store = VectorStore.load(cache_dir)
            print(f"[キャッシュ] 完了（{len(self.store)} 個の断片）\n")
            return

        all_chunks = []
        all_sources = []

        paths = sorted(
            p
            for pattern in ("*.txt", "*.pdf")
            for p in glob.glob(os.path.join(folder, pattern))
        )
        for path in paths:
            filename = os.path.basename(path)
            try:
                text = _read_file(path)
            except Exception as e:
                print(f"[警告] {filename} の読み込みをスキップします: {e}")
                continue
            chunks = split_into_chunks(text)
            all_chunks.extend(chunks)
            all_sources.extend([filename] * len(chunks))
            print(f"  読み込み: {filename}（{len(chunks)}個の断片に分割）")

        if not all_chunks:
            print(f"[警告] 読み込める文書が見つかりませんでした: {folder}")
            return

        print(f"[インデックス作成] {len(all_chunks)}個の断片をベクトル化します...")
        vectors = self.embedder.embed(all_chunks)
        self.store.add(all_chunks, all_sources, vectors)
        print(f"[インデックス作成] 完了（合計 {len(self.store)} 個の断片）\n")

        if cache_dir:
            self.store.save(cache_dir)
            print(f"[キャッシュ] {cache_dir} に保存しました\n")

    def ask(
        self,
        question: str,
        top_k: int = 3,
        show_context: bool = True,
        use_rag: bool = True,
    ) -> dict:
        """質問に回答する"""
        if not use_rag or len(self.store) == 0:
            answer = self.llm.generate(question)
            return {"answer": answer, "context": []}

        query_vector = self.embedder.embed([question])[0]
        results = self.store.search(query_vector, top_k=top_k)

        if not results:
            return {"answer": "資料に記載がありません", "context": []}

        if show_context:
            print("--- 検索で選ばれた断片 ---")
            for i, r in enumerate(results, 1):
                print(f"[{i}] 類似度={r['score']:.3f}  出典={r['source']}")
                print(f"    {r['chunk'][:60]}...")
            print("--------------------------\n")

        context_text = "\n\n".join(
            f"（出典: {r['source']}）\n{r['chunk']}" for r in results
        )
        prompt = (
            "以下の参考情報だけを根拠にして、質問に答えてください。\n"
            "参考情報に答えが書かれていない場合は、推測せず「資料に記載がありません」と答えてください。\n"
            "回答の最後に、根拠とした出典のファイル名を示してください。\n\n"
            f"# 参考情報\n{context_text}\n\n"
            f"# 質問\n{question}"
        )
        system = "あなたは提供された社内文書に基づいて質問に答えるアシスタントです。"
        answer = self.llm.generate(prompt, system=system)
        return {"answer": answer, "context": results}
