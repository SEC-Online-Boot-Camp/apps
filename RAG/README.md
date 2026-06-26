# RAGアプリ

Python で実装した RAG（検索拡張生成）アプリです。
文書検索とLLM回答生成を組み合わせて質問に答えます。
社内規程などの文書を検索対象として想定しています。

Azure上の仮想マシンで動作することを想定しており、LLMはAzure OpenAIまたはOpenAI APIを利用できます。

---

## 構成

```none
rag-app/
├── documents/            検索対象の社内規程（4文書）
├── embedder.py           埋め込みバックエンド（openai / azure）
├── llm_client.py         生成バックエンド（openai / azure）
├── vector_store.py       ベクトルストア（コサイン類似度検索）
├── rag.py                RAGパイプライン本体
├── main.py               CLIエントリーポイント
├── app.py                FlaskによるWebアプリ
├── templates/index.html  Web UI
├── requirements.txt      依存ライブラリ
└── .env.example          設定例
```

処理の流れ:

```none
文書 → チャンク分割 → 埋め込み → 保存
                                  ↓
質問 → 埋め込み → 類似検索 → 関連断片を取得 → LLMで回答生成
```

---

## セットアップ

```bash
sudo apt install -y python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

APIキーなどの設定を`.env`に記載してください。

### LLMの切り替え

`.env`の`LLM_PROVIDER`を変えるだけで、呼び出すLLMを切り替えられます。
（埋め込みモデルと生成AIの両方を切り替えます。）

```none
LLM_PROVIDER=azure    # Azure OpenAI（既定）
LLM_PROVIDER=openai   # OpenAI API
```

---

## 実行

### CLI（コマンドライン）

```bash
# 対話モード（空行または Ctrl+C で終了）
python main.py

# 1問モード
python main.py "リフレッシュ休暇は何日もらえますか？"
```


### Webアプリ

```bash
python app.py
```

ブラウザで <http://localhost:5000> を開く。

以上
