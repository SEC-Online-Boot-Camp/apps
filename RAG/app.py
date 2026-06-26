import os
import shutil
import threading

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from rag import RAG

load_dotenv()

app = Flask(__name__)

_DEFAULT_DOCS_DIR = os.environ.get("DOCS_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "documents"
)
_rag_enabled = True
_loaded_folder: str | None = None
_lock = threading.Lock()  # load/reset の競合防止

print("RAG を初期化しています...")
_rag = RAG()
print("準備完了。http://localhost:5000 を開いてください。\n")


@app.route("/")
def index():
    return render_template("index.html", default_docs_dir=_DEFAULT_DOCS_DIR)


@app.route("/load", methods=["POST"])
def load():
    global _rag_enabled, _loaded_folder
    data = request.get_json(silent=True) or {}
    folder = (data.get("folder") or "").strip() or _DEFAULT_DOCS_DIR
    folder = os.path.realpath(folder)  # パストラバーサル対策
    if not os.path.isdir(folder):
        return jsonify({"error": f"フォルダが見つかりません: {folder}"}), 400
    cache_dir = os.path.join(folder, ".cache")
    with _lock:
        _rag.load_documents(folder, cache_dir=cache_dir)
        _rag_enabled = True
        _loaded_folder = folder
    return jsonify({"chunks": len(_rag.store)})


@app.route("/reset", methods=["POST"])
def reset():
    global _rag_enabled, _loaded_folder
    with _lock:
        _rag.reset()
        if _loaded_folder:
            cache_dir = os.path.join(_loaded_folder, ".cache")
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir)
        _rag_enabled = True
        _loaded_folder = None
    return jsonify({"message": "リセットしました"})


@app.route("/chunks", methods=["GET"])
def chunks():
    limit = request.args.get("limit", type=int)
    items = _rag.store.items()
    if limit is not None and limit > 0:
        items = items[:limit]
    return jsonify({"chunks": items})


@app.route("/rag-toggle", methods=["POST"])
def rag_toggle():
    global _rag_enabled
    with _lock:
        _rag_enabled = not _rag_enabled
    return jsonify({"rag_enabled": _rag_enabled})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "質問が空です"}), 400
    result = _rag.ask(question, show_context=False, use_rag=_rag_enabled)
    return jsonify(result)


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", debug=debug)
