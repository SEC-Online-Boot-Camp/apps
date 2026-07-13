import os
from abc import ABC, abstractmethod


def _require_env(name: str) -> str:
    """環境変数の読み込み"""
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name}が .envに設定されていません。.env.example を参考に記入してください。")
    return value


class Embedder(ABC):
    """埋め込みモデルの共通インターフェース"""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """文章のリストを受け取り、ベクトルのリストを返す。"""
        ...


class OpenAIEmbedder(Embedder):
    """OpenAI API 埋め込みモデル"""

    def __init__(self):
        from openai import OpenAI

        self._client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
        self._model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        print(f"[埋め込み] OpenAI を使用: モデル={self._model}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = self._client.embeddings.create(model=self._model, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as e:
            raise RuntimeError(f"埋め込みの生成に失敗しました: {e}") from e


class AzureEmbedder(Embedder):
    """Azure OpenAI 埋め込みモデル"""

    def __init__(self):
        from openai import OpenAI

        endpoint = _require_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
        if not endpoint.endswith("/openai/v1"):
            endpoint += "/openai/v1"

        self._client = OpenAI(
            api_key=_require_env("AZURE_OPENAI_API_KEY"),
            base_url=endpoint,
        )
        self._model = _require_env("AZURE_OPENAI_EMBED_DEPLOYMENT")
        print("[埋め込み] Azure AI Foundry を使用")
        print(f"  Endpoint : {endpoint}")
        print(f"  Deployment : {self._model}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = self._client.embeddings.create(model=self._model, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as e:
            raise RuntimeError(f"埋め込みの生成に失敗しました: {e}") from e


def create_embedder() -> Embedder:
    """埋め込みモデル生成"""
    if os.environ.get("LLM_PROVIDER", "azure").lower() == "azure":
        return AzureEmbedder()
    return OpenAIEmbedder()
