import os
from abc import ABC, abstractmethod

_LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))


def _require_env(name: str) -> str:
    """環境変数の読み込み"""

    value = os.environ.get(name)
    if not value:
        raise ValueError(
            f"{name} が .env に設定されていません。.env.example を参考に記入してください。"
        )
    return value


class LLMClient(ABC):
    """生成LLMの共通インターフェース"""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str: ...


class _OpenAICompatibleClient(LLMClient):
    """OpenAI互換APIクライアント"""

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=_LLM_TEMPERATURE,
            )
            return resp.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"LLM の呼び出しに失敗しました: {e}") from e


class AzureOpenAIClient(_OpenAICompatibleClient):
    """Azure OpenAI Serviceクライアント"""

    def __init__(self):
        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
            api_key=_require_env("AZURE_OPENAI_API_KEY"),
        )
        self._model = _require_env("AZURE_OPENAI_CHAT_DEPLOYMENT")
        print(f"[生成] Azure OpenAI を使用: デプロイ名={self._model}")


class OpenAIClient(_OpenAICompatibleClient):
    """OpenAI API クライアント"""

    def __init__(self):
        from openai import OpenAI

        self._client = OpenAI(
            api_key=_require_env("OPENAI_API_KEY"),
        )
        self._model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        print(f"[生成] OpenAI を使用: モデル={self._model}")


def create_llm() -> LLMClient:
    """LLMクライアント生成"""
    backend = os.environ.get("LLM_PROVIDER", "azure").lower()
    if backend == "azure":
        return AzureOpenAIClient()
    return OpenAIClient()
