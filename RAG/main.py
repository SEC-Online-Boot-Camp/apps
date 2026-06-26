import argparse
import os

from dotenv import load_dotenv
from rag import RAG

load_dotenv()

_DEFAULT_DOCS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "documents"
)


def main():
    parser = argparse.ArgumentParser(description="社内規程RAG")
    parser.add_argument(
        "-d",
        "--docs",
        default=_DEFAULT_DOCS_DIR,
        metavar="フォルダ",
        help="読み込む文書フォルダ（既定: documents/）",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        metavar="質問",
        help="質問（省略すると対話モード）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RAG 起動")
    print("=" * 60)

    cache_dir = os.path.join(args.docs, ".cache")
    rag = RAG()
    print(f"\n[文書の読み込み] フォルダ: {args.docs}")
    rag.load_documents(args.docs, cache_dir=cache_dir)

    if args.question:
        # 1問モード
        print("=" * 60)
        print(f"Q. {args.question}\n")
        result = rag.ask(args.question)
        print(f"A. {result['answer']}\n")
    else:
        # 対話モード: 空行または Ctrl+C で終了
        print("質問を入力してください（終了: 空行で Enter または Ctrl+C）\n")
        while True:
            try:
                question = input("Q. ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n終了します。")
                break
            if not question:
                break
            print()
            result = rag.ask(question)
            print(f"A. {result['answer']}\n")


if __name__ == "__main__":
    main()
